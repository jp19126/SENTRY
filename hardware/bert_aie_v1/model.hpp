#pragma once
// Complete L=256 BERT-Mini path. Supporting engineering, not a novelty claim.
// Tensor ordering is specified in parameter_layout.json / generated exporter.
#include <cstdint>
#include <cmath>
#include <cfloat>
#ifdef BERT_HLS
#include <hls_math.h>
#define BERT_MATH hls
#else
#define BERT_MATH
#endif
namespace bert_aie {
constexpr int N=256,D=256,F=1024,HEAD=64,LAYERS=4,VOCAB=30522;
constexpr int HIDDEN=N*D, FF=N*F;
constexpr int WM=32,WK=256,WN=32;
constexpr int AM=32,AK=64,AN=32;
constexpr int WORD=0, POSITION=WORD+VOCAB*D, TYPE=POSITION+512*D;
constexpr int EMB_G=TYPE+2*D, EMB_B=EMB_G+D, LAYER_P=EMB_B+D;
// q,k,v,o biases; norm1 gamma/beta; ff1/ff2 biases; norm2 gamma/beta.
constexpr int LP=4*D+2*D+F+D+2*D;
constexpr int POOL_W=LAYER_P+LAYERS*LP,POOL_B=POOL_W+D*D;
constexpr int CLASS_W=POOL_B+D,CLASS_B=CLASS_W+2*D,PARAM_COUNT=CLASS_B+2;
constexpr int LAYER_WEIGHTS=4*D*D+D*F+F*D;
constexpr int LAYER_SCALES=5*D+F;
constexpr int SCRATCH_FLOATS=7*HIDDEN+FF;
inline float bf16(float v) {
    union {float f; uint32_t u;} x; x.f=v;
    // Round-to-nearest-even; preserve NaN rather than overflowing into infinity.
    if ((x.u&0x7fffffffU)>0x7f800000U) x.u|=0x00400000U;
    else x.u+=0x7fffU+((x.u>>16)&1U);
    x.u&=0xffff0000U; return x.f;
}
inline int8_t quant(float x,float scale) {
    float v=x/scale;
    v=v>127.f?127.f:(v< -127.f?-127.f:v);
    float f=BERT_MATH::floorf(v),r=v-f;
    int q=(int)f;
    if(r>0.5f || (r==0.5f && (q&1))) ++q;
    return (int8_t)q;
}
inline void norm(float* x,const float* g,const float* b,int rows) {
    for(int r=0;r<rows;++r) {
        float mean=0.f,var=0.f;
        for(int c=0;c<D;++c) mean+=x[r*D+c];
        mean/=D;
        for(int c=0;c<D;++c) {float v=x[r*D+c]-mean;var+=v*v;}
        float inv=1.f/BERT_MATH::sqrtf(var/D+1.e-12f);
        for(int c=0;c<D;++c) x[r*D+c]=(x[r*D+c]-mean)*inv*g[c]+b[c];
    }
}
// W8 service32x256x32 and BF16 service32x64x32; both native4-row microtiles.
// Wire adapter converts A to native micro-tile order; B is already compatible.
struct CpuTiles {
    uint64_t integer_tiles=0,attention_tiles=0;
    void i8(const int8_t* a,const int8_t* b,int32_t* c) {
        ++integer_tiles;
        for(int i=0;i<WM;++i) for(int j=0;j<WN;++j) {
            int32_t s=0;for(int k=0;k<WK;++k)s+=int(a[i*WK+k])*int(b[k*WN+j]);c[i*WN+j]=s;
        }
    }
    void bfloat(const float* a,const float* b,float* c) {
        ++attention_tiles;
        for(int i=0;i<AM;++i) for(int j=0;j<AN;++j) {
            float s=0;for(int k=0;k<AK;++k)s+=bf16(a[i*AK+k])*bf16(b[k*AN+j]);c[i*AN+j]=s;
        }
    }
};
template<class Tiles> void linear(Tiles& engine,const float* x,const int8_t* w,
        const float* scales,float input_scale,const float* bias,float* out,int K,int O) {
    // Full quantized activation reused across output panels, not re-quantized per tile.
    static int8_t codes[FF];
    for(int i=0;i<N*K;++i) codes[i]=quant(x[i],input_scale);
    for(int r=0;r<N;r+=WM) for(int o=0;o<O;o+=WN) {
        int32_t accum[WM*WN]={};
        for(int k=0;k<K;k+=WK) {
            int8_t a[WM*WK],b[WK*WN];int32_t c[WM*WN];
            for(int i=0;i<WM;++i) for(int t=0;t<WK;++t)a[i*WK+t]=codes[(r+i)*K+k+t];
            // Contiguous DDR reads within each source output row; transpose in PL.
            for(int j=0;j<WN;++j)for(int t=0;t<WK;++t)b[t*WN+j]=w[(o+j)*K+k+t];
            engine.i8(a,b,c);
            for(int i=0;i<WM*WN;++i)accum[i]+=c[i];
        }
        for(int i=0;i<WM;++i)for(int j=0;j<WN;++j)
            out[(r+i)*O+o+j]=(float(accum[i*WN+j])*input_scale)*scales[o+j]+bias[o+j];
    }
}
template<class Tiles> void attention(Tiles& e,const float* q,const float* k,
        const float* v,const int32_t* mask,float* out) {
    for(int h=0;h<4;++h)for(int r=0;r<N;r+=AM) {
        float scores[AM*N];
        for(int col=0;col<N;col+=AN) {
            float a[AM*AK],b[AK*AN],c[AM*AN];
            for(int i=0;i<AM;++i)for(int t=0;t<AK;++t)a[i*AK+t]=q[(r+i)*D+h*HEAD+t];
            for(int j=0;j<AN;++j)for(int t=0;t<AK;++t)b[t*AN+j]=k[(col+j)*D+h*HEAD+t];
            e.bfloat(a,b,c);
            for(int i=0;i<AM;++i)for(int j=0;j<AN;++j)scores[i*N+col+j]=c[i*AN+j]*0.125f+(mask[col+j]?0.f:-FLT_MAX);
        }
        for(int i=0;i<AM;++i) {
            float m=-FLT_MAX,sum=0.f;
            for(int j=0;j<N;++j)if(scores[i*N+j]>m)m=scores[i*N+j];
            for(int j=0;j<N;++j){float s=BERT_MATH::expf(scores[i*N+j]-m);scores[i*N+j]=s;sum+=s;}
            for(int j=0;j<N;++j)scores[i*N+j]/=sum;
        }
        for(int col=0;col<HEAD;col+=AN) {
            float accum[AM*AN]={};
            for(int kb=0;kb<N;kb+=64) {
                float a[AM*AK],b[AK*AN],c[AM*AN];
                for(int i=0;i<AM;++i)for(int t=0;t<AK;++t)a[i*AK+t]=scores[i*N+kb+t];
                for(int t=0;t<AK;++t)for(int j=0;j<AN;++j)b[t*AN+j]=v[(kb+t)*D+h*HEAD+col+j];
                e.bfloat(a,b,c);
                for(int i=0;i<AM*AN;++i)accum[i]+=c[i];
            }
            for(int i=0;i<AM;++i)for(int j=0;j<AN;++j)out[(r+i)*D+h*HEAD+col+j]=accum[i*AN+j];
        }
    }
}
template<bool Resident,class Tiles> int model(Tiles& e,const int32_t* ids,const int32_t* mask,
        const int32_t* types,const float* p,const int8_t* w,const float* scales,
        const float* input_scales,float* scratch,float* logits) {
    // Conventional mapping keeps full intermediate tensors in DDR, but uses the
    // same on-chip per-operator tiles and quantized activation reuse as resident.
    static float resident[Resident?SCRATCH_FLOATS:1];
#pragma HLS BIND_STORAGE variable=resident type=ram_2p impl=uram
    float* storage=Resident?resident:scratch;
    float* x=storage;float* q=x+HIDDEN;float* k=q+HIDDEN;float* v=k+HIDDEN;
    float* a=v+HIDDEN;float* y=a+HIDDEN;float* z=y+HIDDEN;float* ff=z+HIDDEN;
    int valid_keys=0;
    for(int i=0;i<N;++i) {
        if(ids[i]<0||ids[i]>=VOCAB||types[i]<0||types[i]>=2||(mask[i]!=0&&mask[i]!=1))return 1;
        valid_keys+=mask[i];
        for(int j=0;j<D;++j)x[i*D+j]=(p[WORD+ids[i]*D+j]+p[TYPE+types[i]*D+j])+p[POSITION+i*D+j];
    }
    if(valid_keys==0)return 2;
    norm(x,p+EMB_G,p+EMB_B,N);
    for(int l=0;l<LAYERS;++l) {
        const float* b=p+LAYER_P+l*LP;const int8_t* lw=w+l*LAYER_WEIGHTS;
        const float* s=scales+l*LAYER_SCALES;const float* is=input_scales+6*l;
        linear(e,x,lw,s,is[0],b,q,D,D);
        linear(e,x,lw+D*D,s+D,is[1],b+D,k,D,D);
        linear(e,x,lw+2*D*D,s+2*D,is[2],b+2*D,v,D,D);
        attention(e,q,k,v,mask,a);
        linear(e,a,lw+3*D*D,s+3*D,is[3],b+3*D,y,D,D);
        for(int i=0;i<HIDDEN;++i)y[i]+=x[i];
        norm(y,b+4*D,b+5*D,N);
        linear(e,y,lw+4*D*D,s+4*D,is[4],b+6*D,ff,D,F);
        for(int i=0;i<FF;++i)ff[i]=0.5f*ff[i]*(1.f+BERT_MATH::erff(ff[i]*0.7071067811865475244f));
        linear(e,ff,lw+4*D*D+D*F,s+4*D+F,is[5],b+6*D+F,z,F,D);
        for(int i=0;i<HIDDEN;++i)x[i]=z[i]+y[i];
        norm(x,b+7*D+F,b+8*D+F,N);
    }
    float pooled[D];
    for(int o=0;o<D;++o) {
        float sum=0.f;for(int k=0;k<D;++k)sum+=x[k]*p[POOL_W+o*D+k];
        pooled[o]=BERT_MATH::tanhf(sum+p[POOL_B+o]);
    }
    for(int o=0;o<2;++o) {
        float sum=0.f;for(int k=0;k<D;++k)sum+=pooled[k]*p[CLASS_W+o*D+k];
        logits[o]=sum+p[CLASS_B+o];
    }
    return 0;
}
}
