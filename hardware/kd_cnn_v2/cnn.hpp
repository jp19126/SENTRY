#ifndef SENTRY_KD_CNN_HPP
#define SENTRY_KD_CNN_HPP
#include <cstdint>
namespace kdcnn {
constexpr int V=30522,C=64,L=256,D=6,K=3,MAX_WINDOWS=16;
constexpr uint32_t MAGIC=0x4b444332,VERSION=1;
constexpr int PARAM_BYTES=32+V*C+C*C+C*8+C*4+D*(C*C*K+C*8+C*4+4)+2*(2*C)+2*8+2*4;
struct Model {
 int8_t embedding[V][C], projection[C][C], conv[D][C][C][K], head[2][2*C];
 int64_t projection_bias[C],bias[D][C],head_bias[2];
 int32_t projection_shift[C],shift[D][C],skip_shift[D],head_shift[2];
};
struct Result {int32_t margin, maximum, blocked, error;uint32_t windows,decision_valid;};
int64_t rne_shift(int64_t x,int shift,int64_t cap);
int64_t rne_div(int64_t x,int32_t divisor);
bool load_model(const uint8_t*,uint32_t,Model&);
int32_t window(const Model&,const uint16_t*,const uint8_t*,int32_t&);
}
extern "C" void kd_cnn_top(const uint8_t *blob,uint32_t bytes,const uint16_t *ids,
 const uint8_t *masks,uint32_t count,uint32_t load,uint32_t first,uint32_t final,
 int32_t previous,uint32_t previous_windows,int32_t threshold,kdcnn::Result *output);
#endif
