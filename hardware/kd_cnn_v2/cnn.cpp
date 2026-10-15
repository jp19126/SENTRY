#include "cnn.hpp"
#include <climits>
#ifndef CNN_LANES
#define CNN_LANES 16
#endif
static_assert(CNN_LANES==4 || CNN_LANES==16,"supported lane controls");
namespace kdcnn {
constexpr int64_t CAP=int64_t(1)<<30;
int64_t rne_shift(int64_t x,int shift,int64_t cap) {
 if(shift>=0){if(shift>=62)return x>0?cap:x<0?-cap:0;const int64_t f=int64_t(1)<<shift;
  if(x>cap/f)return cap;if(x< -cap/f)return -cap;return x*f;}
 const unsigned s=unsigned(-shift);if(s>=64)return 0;
 const uint64_t a=x<0?uint64_t(-(x+1))+1:uint64_t(x),q=a>>s;
 const uint64_t r=a&((uint64_t(1)<<s)-1),h=uint64_t(1)<<(s-1);
 const uint64_t v=q+uint64_t(r>h||(r==h&&(q&1)));
 const int64_t z=x<0?-int64_t(v):int64_t(v);return z>cap?cap:z< -cap?-cap:z;
}
int64_t rne_div(int64_t x,int32_t d){
 const int64_t a=x<0?-x:x,q=a/d,r=a%d;const int64_t y=q+int64_t(2*r>d||(2*r==d&&(q&1)));return x<0?-y:y;
}
template<class T> static T read(const uint8_t *p,uint32_t& pos){
#pragma HLS INLINE
 uint64_t x=0;for(unsigned b=0;b<sizeof(T);++b)x|=uint64_t(p[pos++])<<(8*b);return T(x);
}
bool load_model(const uint8_t *p,uint32_t n,Model& m){
 if(n!=PARAM_BYTES)return false;uint32_t pos=0,h[8];for(int i=0;i<8;++i)h[i]=read<uint32_t>(p,pos);
 if(h[0]!=MAGIC||h[1]!=VERSION||h[2]!=V||h[3]!=C||h[4]!=L||h[5]!=D||h[6]!=12||h[7]!=PARAM_BYTES)return false;
 for(int v=0;v<V;++v)for(int c=0;c<C;++c)m.embedding[v][c]=read<int8_t>(p,pos);
 for(int o=0;o<C;++o)for(int c=0;c<C;++c)m.projection[o][c]=read<int8_t>(p,pos);
 for(int o=0;o<C;++o)m.projection_bias[o]=read<int64_t>(p,pos);
 for(int o=0;o<C;++o)m.projection_shift[o]=read<int32_t>(p,pos);
 for(int d=0;d<D;++d){
  for(int o=0;o<C;++o)for(int c=0;c<C;++c)for(int k=0;k<K;++k)m.conv[d][o][c][k]=read<int8_t>(p,pos);
  for(int o=0;o<C;++o)m.bias[d][o]=read<int64_t>(p,pos);
  for(int o=0;o<C;++o)m.shift[d][o]=read<int32_t>(p,pos);
  m.skip_shift[d]=read<int32_t>(p,pos);
 }
 for(int o=0;o<2;++o)for(int c=0;c<2*C;++c)m.head[o][c]=read<int8_t>(p,pos);
 for(int o=0;o<2;++o)m.head_bias[o]=read<int64_t>(p,pos);
 for(int o=0;o<2;++o)m.head_shift[o]=read<int32_t>(p,pos);
 // Export contract bounds prevent overflow in exact signed accumulations.
 for(int o=0;o<C;++o)if(m.projection_shift[o]<-62||m.projection_shift[o]>31||m.projection_bias[o]> (int64_t(1)<<55)||m.projection_bias[o]< -(int64_t(1)<<55))return false;
 for(int d=0;d<D;++d){if(m.skip_shift[d]<-62||m.skip_shift[d]>31)return false;
  for(int o=0;o<C;++o)if(m.shift[d][o]<-62||m.shift[d][o]>31||m.bias[d][o]>(int64_t(1)<<55)||m.bias[d][o]< -(int64_t(1)<<55))return false;}
 for(int o=0;o<2;++o)if(m.head_shift[o]<-62||m.head_shift[o]>31||m.head_bias[o]>(int64_t(1)<<55)||m.head_bias[o]< -(int64_t(1)<<55))return false;
 return pos==n;
}
static void project(const Model& m,const int8_t in[L][C],const uint8_t mask[L],int8_t out[L][C]){
 for(int t=0;t<L;++t)for(int o=0;o<C;++o){int32_t a=0;
  for(int cb=0;cb<C;cb+=CNN_LANES){
#pragma HLS PIPELINE II=1
   int32_t part=0;for(int c=0;c<CNN_LANES;++c){
#pragma HLS UNROLL
    part+=int32_t(in[t][cb+c])*int32_t(m.projection[o][cb+c]);
   }a+=part;
  }out[t][o]=mask[t]?int8_t(rne_shift(int64_t(a)+m.projection_bias[o],m.projection_shift[o],127)):0;
 }
}
static void block(const Model& m,int d,const int8_t in[L][C],const uint8_t mask[L],int8_t out[L][C]){
 const int dilation=1<<d;
 for(int t=0;t<L;++t)for(int o=0;o<C;++o){int32_t a=0;
  for(int k=0;k<K;++k)for(int cb=0;cb<C;cb+=CNN_LANES){
#pragma HLS PIPELINE II=1
   const int at=t+(k-1)*dilation;int32_t part=0;
   for(int c=0;c<CNN_LANES;++c){
#pragma HLS UNROLL
    const int8_t v=at>=0&&at<L?in[at][cb+c]:0;
    part+=int32_t(v)*int32_t(m.conv[d][o][cb+c][k]);
   }a+=part;
  }
  const int64_t branch=rne_shift(int64_t(a)+m.bias[d][o],m.shift[d][o],CAP);
  const int64_t skip=rne_shift(in[t][o],m.skip_shift[d],CAP);
  const int64_t joined=skip+(branch>0?branch:0);
  out[t][o]=mask[t]?int8_t(joined>127?127:joined< -127?-127:joined):0;
 }
}
int32_t window(const Model& m,const uint16_t *ids,const uint8_t *mask,int32_t& error){
 int8_t a[L][C],b[L][C];uint8_t valid[L];
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=CNN_LANES dim=2
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=CNN_LANES dim=2
#pragma HLS BIND_STORAGE variable=a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b type=ram_2p impl=bram
 int32_t count=0;error=0;
 for(int t=0;t<L;++t){valid[t]=mask[t];if(valid[t]>1||(valid[t]&&ids[t]>=V)){error=2;return 0;}count+=valid[t];
  for(int c=0;c<C;++c){
#pragma HLS PIPELINE II=1
   a[t][c]=valid[t]?m.embedding[ids[t]][c]:0;
  }
 }
 project(m,a,valid,b);
 for(int d=0;d<D;++d){if(d%2==0)block(m,d,b,valid,a);else block(m,d,a,valid,b);}
 // D is even: b contains final layer. Max excludes padding, empty => zero.
 int8_t pool[2*C];
 for(int c=0;c<C;++c){int32_t sum=0;int8_t mx=-127;
  for(int t=0;t<L;++t){
#pragma HLS PIPELINE II=1
   if(valid[t]){sum+=b[t][c];if(b[t][c]>mx)mx=b[t][c];}
  }pool[c]=count?mx:0;pool[C+c]=count?int8_t(rne_div(sum,count)):0;
 }
 int32_t logits[2];for(int o=0;o<2;++o){int32_t dot=0;for(int c=0;c<2*C;++c){
#pragma HLS PIPELINE II=1
   dot+=int32_t(pool[c])*int32_t(m.head[o][c]);
  }logits[o]=int32_t(rne_shift(int64_t(dot)+m.head_bias[o],m.head_shift[o],CAP-1));
 }return logits[1]-logits[0];
}
}
extern "C" void kd_cnn_top(const uint8_t *blob,uint32_t bytes,const uint16_t *ids,const uint8_t *masks,uint32_t count,uint32_t load,uint32_t first,uint32_t final,int32_t previous,uint32_t previous_windows,int32_t threshold,kdcnn::Result *output){
#pragma HLS INTERFACE m_axi port=blob offset=slave bundle=model depth=2040000
#pragma HLS INTERFACE m_axi port=ids offset=slave bundle=input depth=4096
#pragma HLS INTERFACE m_axi port=masks offset=slave bundle=input depth=4096
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=output depth=1
#pragma HLS INTERFACE s_axilite port=return bundle=control
 using namespace kdcnn;static Model model;static bool ready=false;
#pragma HLS DISAGGREGATE variable=model
#pragma HLS BIND_STORAGE variable=model.embedding type=ram_2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=model.embedding cyclic factor=CNN_LANES dim=2
#pragma HLS ARRAY_PARTITION variable=model.projection cyclic factor=CNN_LANES dim=2
#pragma HLS ARRAY_PARTITION variable=model.conv cyclic factor=CNN_LANES dim=3
 Result r={0,first?INT_MIN:previous,0,0,first?0:previous_windows,0};
 if(load>1||(load&&!first)){r.error=3;*output=r;return;}
 if(load)ready=load_model(blob,bytes,model);
 if(!ready){r.error=1;*output=r;return;}
 if(count==0||count>MAX_WINDOWS||first>1||final>1||(first&&previous_windows)||(!first&&!previous_windows)||previous_windows>UINT_MAX-count){r.error=3;*output=r;return;}
 for(uint32_t w=0;w<count;++w){
#pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=1
  r.margin=window(model,ids+w*L,masks+w*L,r.error);if(r.error)break;
  if(r.margin>r.maximum)r.maximum=r.margin;++r.windows;
 }r.decision_valid=final&&!r.error;r.blocked=r.decision_valid&&(r.maximum>threshold);*output=r;
}
