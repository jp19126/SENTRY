#include "span_guard.hpp"
#ifdef __SYNTHESIS__
#include <hls_stream.h>
#else
#include <vector>
#endif
#ifndef SPAN_MAPPING
#define SPAN_MAPPING 0
#endif
#ifndef SPAN_ALGORITHM
#define SPAN_ALGORITHM 0
#endif
namespace sentry_span {
static Result sopa_impl(const Model&,const uint16_t*,Window,uint32_t,Trace*);
int64_t rne_right(int64_t x,unsigned s) {
    if(!s)return x;
    if(s>=64)return 0;
    const uint64_t m=x<0?uint64_t(-(x+1))+1:uint64_t(x),q=m>>s;
    const uint64_t rem=m&((uint64_t(1)<<s)-1),half=uint64_t(1)<<(s-1);
    const uint64_t y=q+uint64_t(rem>half||(rem==half&&(q&1)));
    return x<0?-int64_t(y):int64_t(y);
}
int64_t scale_pow2(int64_t x,int shift,int64_t cap) {
    if(shift<0){const int64_t y=rne_right(x,unsigned(-shift));return y>cap?cap:(y< -cap?-cap:y);}
    if(shift>=63)return x>0?cap:(x<0?-cap:0);
    const int64_t f=int64_t(1)<<shift;
    if(x>cap/f)return cap;
    if(x< -cap/f)return -cap;
    return x*f;
}
template<class T> T read_scalar(const uint8_t *blob,uint32_t& pos) {
#pragma HLS INLINE
    uint64_t value=0;
    for(unsigned b=0;b<sizeof(T);++b)value|=uint64_t(blob[pos++])<<(8*b);
    return T(value);
}
template<class T,int N> void read_array(const uint8_t *blob,uint32_t& pos,T (&out)[N]) {
#pragma HLS INLINE
    // Static array extent retains field bounds through HLS disaggregation.
    // Scalar fields never decay to a one-element pointer with generic indexing.
    for(int i=0;i<N;++i)out[i]=read_scalar<T>(blob,pos);
}
bool load_model(const uint8_t *blob,uint32_t bytes,Model& m) {
    if(bytes!=BLOB_BYTES)return false;
    uint32_t p=0,h[8];read_array(blob,p,h);
    if(h[0]!=MAGIC||h[1]!=VERSION||h[2]!=SPAN_ALGORITHM||h[3]!=H||h[4]!=R||h[5]!=Z||h[6]!=VOCAB||h[7]!=PAYLOAD_BYTES)return false;
    m.kind=h[2];
    for(int v=0;v<VOCAB;++v)read_array(blob,p,m.emb_w[v]);
    for(int o=0;o<Z;++o)for(int k=0;k<3+2*(o/64);++k)read_array(blob,p,m.conv_w[o][k]);
    read_array(blob,p,m.conv_bias);read_array(blob,p,m.conv_shift);
    for(int r=0;r<2*R;++r)read_array(blob,p,m.proj_w[r]);
    read_array(blob,p,m.proj_bias);read_array(blob,p,m.proj_shift);
    for(int s=0;s<2;++s)read_array(blob,p,m.lambda_q8[s]);
    read_array(blob,p,m.head_a);read_array(blob,p,m.head_b);
    read_array(blob,p,m.head_c);read_array(blob,p,m.head_anchor);
    m.head_bias=read_scalar<int32_t>(blob,p);m.z_exp=read_scalar<int32_t>(blob,p);
    read_array(blob,p,m.sigmoid_bits);
    for(int r=0;r<32;++r)read_array(blob,p,m.mlp1_w[r]);
    read_array(blob,p,m.mlp1_b);read_array(blob,p,m.mlp2_w);
    for(int s=0;s<2;++s)read_array(blob,p,m.epsilon_q12[s]);
    read_array(blob,p,m.pattern_head);
    read_array(blob,p,m.tanh_q12);read_array(blob,p,m.logsigmoid_q12);
    for(int r=0;r<R;++r)if(m.lambda_q8[0][r]>256||m.lambda_q8[1][r]>256)return false;
    for(int j=0;j<Z;++j)if(m.conv_shift[j]<-62||m.conv_shift[j]>31)return false;
    for(int j=0;j<2*R;++j)if(m.proj_shift[j]<-62||m.proj_shift[j]>31)return false;
    if(m.z_exp<-30||m.z_exp>16)return false;
    for(int i=0;i<LUT_SIZE;++i)if(m.sigmoid_bits[i]>0x3f800000u||(i&&m.sigmoid_bits[i]<m.sigmoid_bits[i-1]))return false;
    return p==bytes;
}
uint32_t risk_lookup(const Model& m,int64_t g) {
    g=g>65536?65536:(g< -65536?-65536:g);
    return m.sigmoid_bits[int(rne_right(g,5))+2048];
}
static bool eligible(int j,int n,Window w){return(j>=31||w.left_edge)&&(j+32<n||w.right_edge);}
static bool prepare(const Model& m,const uint16_t *ids,Window w,int8_t x[MAX_TOKENS][EMB]) {
    if(w.length>MAX_TOKENS||w.left_edge>1||w.right_edge>1||w.content_start>w.document_length
        ||w.length>w.document_length-w.content_start||w.left_edge!=(w.content_start==0)
        ||w.right_edge!=(w.content_start+w.length==w.document_length))return false;
    for(int t=0;t<w.length;++t){
#pragma HLS LOOP_TRIPCOUNT min=0 max=254 avg=254
      if(ids[t]>=VOCAB)return false;for(int e=0;e<EMB;++e){
#pragma HLS PIPELINE II=1
        x[t][e]=m.emb_w[ids[t]][e];
    }}return true;
}
static void encode_z(const Model& m,const int8_t x[MAX_TOKENS][EMB],int t,int n,int8_t z[Z]) {
    // Prospective128-product tile; actual scheduling/resources need HLS evidence.
    for(int ob=0;ob<Z;ob+=16){int64_t acc[16];
#pragma HLS ARRAY_PARTITION variable=acc complete
        for(int o=0;o<16;++o)acc[o]=m.conv_bias[ob+o];
        const int br=ob/64,width=3+2*br;
        for(int k=0;k<width;++k){
#pragma HLS LOOP_TRIPCOUNT min=3 max=7 avg=5
          for(int eb=0;eb<EMB;eb+=8){
#pragma HLS PIPELINE II=1
            for(int o=0;o<16;++o){
#pragma HLS UNROLL
                int64_t part=0;
                for(int e=0;e<8;++e){
#pragma HLS UNROLL
                    const int at=t+k-width/2;
                    const int8_t a=at>=0&&at<n?x[at][eb+e]:0;
                    part+=int32_t(a)*int32_t(m.conv_w[ob+o][k][eb+e]);
                }acc[o]+=part;
            }
          }
        }
        for(int o=0;o<16;++o){const int64_t q=scale_pow2(acc[o],m.conv_shift[ob+o],127);z[ob+o]=int8_t(q<0?0:q);}
    }
}
static void project(const Model& m,const int8_t z[Z],int32_t uv[2*R]){
    for(int o=0;o<2*R;++o){int64_t a=m.proj_bias[o];for(int kb=0;kb<Z;kb+=32){
#pragma HLS PIPELINE II=1
        int64_t part=0;for(int k=0;k<32;++k){
#pragma HLS UNROLL
            part+=int32_t(z[kb+k])*int32_t(m.proj_w[o][kb+k]);
        }a+=part;
    }uv[o]=int32_t(scale_pow2(a,m.proj_shift[o],SPAN_ALGORITHM==3?65536:32768));}
}
static void encode(const Model& m,const int8_t x[MAX_TOKENS][EMB],int t,int n,int8_t z[Z],int32_t uv[2*R]){encode_z(m,x,t,n,z);project(m,z,uv);}
static void update(const Model& m,int32_t a[R],int32_t b[R],const int32_t uv[2*R],bool valid=true) {
    for(int rb=0;rb<R;rb+=8){
#pragma HLS PIPELINE II=1
      for(int e=0;e<8;++e){
#pragma HLS UNROLL
        const int r=rb+e;
        const int64_t aa=rne_right(int64_t(a[r])*m.lambda_q8[0][r],8)+(valid?uv[r]:0);
        const int64_t bb=rne_right(int64_t(b[r])*m.lambda_q8[1][r],8)+(valid?uv[R+r]:0);
        a[r]=int32_t(aa>32768?32768:(aa< -32768?-32768:aa));
        b[r]=int32_t(bb>32768?32768:(bb< -32768?-32768:bb));
      }
    }
}
static int64_t head_score(const Model& m,const int32_t a[R],const int32_t b[R],const int8_t z[Z]) {
    int64_t state=0,anchor=0;
    for(int r=0;r<R;++r){
#pragma HLS PIPELINE II=1
        const int64_t ab=rne_right(int64_t(a[r])*b[r],12);
        state+=int64_t(a[r])*m.head_a[r]+int64_t(b[r])*m.head_b[r]+ab*m.head_c[r];
    }
    for(int jb=0;jb<Z;jb+=32){
#pragma HLS PIPELINE II=1
        int64_t part=0;for(int k=0;k<32;++k){
#pragma HLS UNROLL
            part+=int64_t(z[jb+k])*m.head_anchor[jb+k];
        }anchor+=part;
    }
    if(SPAN_ALGORITHM==2){
        state=0;
        for(int o=0;o<32;++o){int64_t value=0;
            for(int r=0;r<R;++r)value+=int64_t(a[r])*m.mlp1_w[o][r]+int64_t(b[r])*m.mlp1_w[o][R+r];
            value=rne_right(value,12)+m.mlp1_b[o];
            if(value>0)state+=value*m.mlp2_w[o];
        }
    }
    return rne_right(state,12)+scale_pow2(anchor,m.z_exp,int64_t(1)<<60)+m.head_bias;
}
static Result reference_impl(const Model& m,const uint16_t *ids,Window w,uint32_t threshold,Trace *trace) {
    Result result={0,0,OK,0,1};int8_t x[MAX_TOKENS][EMB],zs[MAX_TOKENS][Z];int32_t proj[MAX_TOKENS][2*R];
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=2
    if(!prepare(m,ids,w,x)){result.error=BAD_WINDOW;return result;}
    for(int t=0;t<w.length;++t){
#pragma HLS LOOP_TRIPCOUNT min=0 max=254 avg=254
      encode(m,x,t,w.length,zs[t],proj[t]);
        if(trace){for(int k=0;k<Z;++k)trace->z[t][k]=zs[t][k];for(int k=0;k<2*R;++k)trace->p[t][k]=proj[t][k];
            trace->eligible[t]=eligible(t,w.length,w);trace->g[t]=-(int64_t(1)<<40);}}
    for(int j=0;j<w.length;++j){
#pragma HLS LOOP_TRIPCOUNT min=0 max=254 avg=254
      if(eligible(j,w.length,w)){
        int32_t a[R]={},b[R]={};
        for(int p=j-28;p<=j+29;++p){
            // One array base; padded sites select zeros inside the update.
            const bool valid=p>=0&&p<w.length;
            update(m,a,b,proj[valid?p:0],valid);
        }
        const int64_t g=head_score(m,a,b,zs[j]);const uint32_t risk=risk_lookup(m,g);
        if(trace){for(int r=0;r<R;++r){trace->a[j][r]=a[r];trace->b[j][r]=b[r];}trace->g[j]=g;}
        result.risk_bits=risk>result.risk_bits?risk:result.risk_bits;++result.anchors;
      }
    }result.blocked=result.risk_bits>threshold;return result;
}
Result window_reference(const Model& m,const uint16_t *ids,Window w,uint32_t t){return reference_impl(m,ids,w,t,nullptr);}
Result window_trace(const Model& m,const uint16_t *ids,Window w,uint32_t t,Trace& trace){return SPAN_ALGORITHM==3?sopa_impl(m,ids,w,t,&trace):reference_impl(m,ids,w,t,&trace);}
struct Token {int16_t position;int8_t z[Z];int32_t uv[2*R];};
#ifdef __SYNTHESIS__
using TokenStream=hls::stream<Token>;
#else
// Sequential C simulation is unbounded; RTL FIFO depths are declared below.
struct TokenStream {std::vector<Token> values;size_t cursor=0;void write(const Token& t){values.push_back(t);}Token read(){return values.at(cursor++);}};
#endif
static void encode_stream(const Model& m,const int8_t x[MAX_TOKENS][EMB],int n,TokenStream& out){
    for(int p=-28;p<n+29;++p){
#pragma HLS LOOP_TRIPCOUNT min=57 max=311 avg=311
      Token token={};token.position=int16_t(p);
        if(p>=0&&p<n)encode_z(m,x,p,n,token.z);out.write(token);}
}
static void projection_stream(const Model& m,int n,TokenStream& in,TokenStream& out){
    for(int t=0;t<n+57;++t){
#pragma HLS LOOP_TRIPCOUNT min=57 max=311 avg=311
      Token token=in.read();if(token.position>=0&&token.position<n)project(m,token.z,token.uv);out.write(token);}
}
static void state_stream(const Model& m,Window w,uint32_t threshold,TokenStream& in,Result& out){
    Result result={0,0,OK,0,1};int8_t anchor[H][Z];int32_t a[H][R],b[H][R];
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=anchor cyclic factor=32 dim=2
    for(int p=-28;p<int(w.length)+29;++p){
#pragma HLS LOOP_TRIPCOUNT min=57 max=311 avg=311
        const Token token=in.read();
        if(p>=0&&p<w.length&&eligible(p,w.length,w))for(int k=0;k<Z;++k)anchor[p%H][k]=token.z[k];
        const int first=p-29>0?p-29:0,last=p+28<int(w.length)?p+28:int(w.length)-1;
        for(int j=first;j<=last;++j){
#pragma HLS LOOP_TRIPCOUNT min=0 max=58 avg=58
          if(eligible(j,w.length,w)){
            const int lane=j%H;
            if(p==j-28)for(int r=0;r<R;++r){a[lane][r]=0;b[lane][r]=0;}
            update(m,a[lane],b[lane],token.uv);
            if(p==j+29){const uint32_t risk=risk_lookup(m,head_score(m,a[lane],b[lane],anchor[lane]));
                result.risk_bits=risk>result.risk_bits?risk:result.risk_bits;++result.anchors;}
          }
        }
    }result.blocked=result.risk_bits>threshold;out=result;
}
static void pipeline(const Model& m,const int8_t x[MAX_TOKENS][EMB],Window w,uint32_t threshold,Result& result){
#pragma HLS DATAFLOW
    TokenStream encoded,projected;
#pragma HLS STREAM variable=encoded depth=2
#pragma HLS STREAM variable=projected depth=2
    encode_stream(m,x,w.length,encoded);projection_stream(m,w.length,encoded,projected);
    state_stream(m,w,threshold,projected,result);
}
Result window_streamed(const Model& m,const uint16_t *ids,Window w,uint32_t threshold){
    int8_t x[MAX_TOKENS][EMB];
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8 dim=2
    if(!prepare(m,ids,w,x))return Result{0,0,BAD_WINDOW,0,1};Result result;pipeline(m,x,w,threshold,result);return result;
}
static Result sopa_impl(const Model& m,const uint16_t *ids,Window w,uint32_t threshold,Trace *trace){
    Result out={0,0,OK,0,1};int8_t x[MAX_TOKENS][EMB],pooled_z[Z]={};
    if(!prepare(m,ids,w,x)){out.error=BAD_WINDOW;return out;}
    int64_t s1[16],s2[16],pooled[16];
    for(int r=0;r<16;++r){s1[r]=m.epsilon_q12[0][r];s2[r]=s1[r]+m.epsilon_q12[1][r];pooled[r]=s2[r];}
    for(int t=0;t<w.length;++t){
#pragma HLS LOOP_TRIPCOUNT min=0 max=254 avg=254
      int8_t z[Z];int32_t p[2*R];encode(m,x,t,w.length,z,p);
        if(trace){for(int k=0;k<Z;++k)trace->z[t][k]=z[k];for(int k=0;k<2*R;++k)trace->p[t][k]=p[k];trace->eligible[t]=eligible(t,w.length,w);trace->g[t]=-(int64_t(1)<<40);}
        for(int k=0;k<Z;++k)if(z[k]>pooled_z[k])pooled_z[k]=z[k];
        for(int r=0;r<16;++r){
            const int m1=int(rne_right(p[r],5))+2048,m2=int(rne_right(p[16+r],5))+2048;
            const int l1=int(rne_right(p[32+r],5))+2048,l2=int(rne_right(p[48+r],5))+2048;
            int64_t n1=s1[r]+m.logsigmoid_q12[l1],n2=s2[r]+m.logsigmoid_q12[l2];
            if(m.tanh_q12[m1]>n1)n1=m.tanh_q12[m1];if(m.epsilon_q12[0][r]>n1)n1=m.epsilon_q12[0][r];
            const int64_t advance=s1[r]+m.tanh_q12[m2],epsilon=n1+m.epsilon_q12[1][r];
            if(advance>n2)n2=advance;if(epsilon>n2)n2=epsilon;
            s1[r]=n1;s2[r]=n2;if(n2>pooled[r])pooled[r]=n2;
            if(trace){trace->a[t][r]=int32_t(s1[r]);trace->b[t][r]=int32_t(s2[r]);}
        }
    }
    if(w.length){int64_t core=0,anchor=0;for(int r=0;r<16;++r)core+=pooled[r]*m.pattern_head[r];
        for(int k=0;k<Z;++k)anchor+=int64_t(pooled_z[k])*m.head_anchor[k];
        out.risk_bits=risk_lookup(m,rne_right(core,12)+scale_pow2(anchor,m.z_exp,int64_t(1)<<60)+m.head_bias);out.anchors=w.length;}
    out.blocked=out.risk_bits>threshold;return out;
}
Result window_sopa(const Model& m,const uint16_t *ids,Window w,uint32_t threshold){return sopa_impl(m,ids,w,threshold,nullptr);}
}
extern "C" void span_guard_top(const uint8_t *blob,uint32_t bytes,const uint16_t *ids,
    const sentry_span::Window *windows,uint32_t count,uint32_t load,uint32_t empty,
    uint32_t previous,uint32_t threshold,sentry_span::Result *output) {
#pragma HLS INTERFACE m_axi port=blob offset=slave bundle=model depth=2100000
#pragma HLS INTERFACE m_axi port=ids offset=slave bundle=input depth=16256
#pragma HLS INTERFACE m_axi port=windows offset=slave bundle=input depth=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=output depth=1
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace sentry_span;static Model model;static bool ready=false;
#pragma HLS DISAGGREGATE variable=model
#pragma HLS BIND_STORAGE variable=model.emb_w type=ram_2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=model.emb_w cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=model.conv_w cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=model.conv_w cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=model.conv_w type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=model.proj_w cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=model.head_anchor cyclic factor=32 dim=1
    Result result={previous,0,OK,0,0};
    if(load)ready=load_model(blob,bytes,model);
    if(!ready){result.error=UNINITIALIZED;*output=result;return;}
    if(count>CHUNK||empty>1||(empty&&(count||previous))||previous>0x3f800000u||threshold>0x3f800000u){result.error=BAD_CHUNK;*output=result;return;}
    if(empty)result.risk_bits=risk_lookup(model,model.head_bias);
    for(uint32_t i=0;i<count;++i){
#pragma HLS LOOP_TRIPCOUNT min=0 max=64 avg=1
      const Window w=windows[i];
        if(w.offset>count*MAX_TOKENS||w.length>count*MAX_TOKENS-w.offset){result.error=BAD_WINDOW;break;}
        Result next=SPAN_ALGORITHM==3?window_sopa(model,ids+w.offset,w,threshold):(SPAN_MAPPING==0?window_streamed(model,ids+w.offset,w,threshold):window_reference(model,ids+w.offset,w,threshold));
        if(next.error){result.error=next.error;break;}
        result.risk_bits=next.risk_bits>result.risk_bits?next.risk_bits:result.risk_bits;
        result.anchors+=next.anchors;++result.windows;
    }result.blocked=result.risk_bits>threshold;*output=result;
}
