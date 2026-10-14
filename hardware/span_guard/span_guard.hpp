#ifndef SENTRY_SPAN_GUARD_HPP
#define SENTRY_SPAN_GUARD_HPP
#include <cstdint>
namespace sentry_span {
constexpr int VOCAB=30522, EMB=64, Z=192, R=32, H=58, MAX_TOKENS=254, CHUNK=64;
constexpr int CONV_WEIGHTS=61440, PROJ_WEIGHTS=12288, LUT_SIZE=4097;
constexpr uint32_t MAGIC=0x53504731u, VERSION=1;
// Blob: eight uint32 header fields, then tightly concatenated little-endian
// arrays in this declaration order. No compiler struct padding is serialized.
struct Model {
    uint32_t kind;
    int8_t emb_w[VOCAB][EMB],conv_w[Z][7][EMB];
    int64_t conv_bias[Z]; int32_t conv_shift[Z];
    int8_t proj_w[2*R][Z]; int64_t proj_bias[2*R]; int32_t proj_shift[2*R];
    uint16_t lambda_q8[2][R];
    int32_t head_a[R],head_b[R],head_c[R],head_anchor[Z],head_bias,z_exp;
    uint32_t sigmoid_bits[LUT_SIZE];
    int32_t mlp1_w[32][2*R],mlp1_b[32],mlp2_w[32];
    int32_t epsilon_q12[2][16],pattern_head[16],tanh_q12[LUT_SIZE],logsigmoid_q12[LUT_SIZE];
};
constexpr uint32_t PAYLOAD_BYTES=VOCAB*EMB+CONV_WEIGHTS+Z*8+Z*4+PROJ_WEIGHTS+2*R*8+2*R*4+2*R*2+(3*R+Z)*4+8+LUT_SIZE*4+(32*2*R+32+32+2*16+16+2*LUT_SIZE)*4;
constexpr uint32_t BLOB_BYTES=32+PAYLOAD_BYTES;
struct Window {uint16_t length;uint8_t left_edge,right_edge;uint32_t offset,content_start,document_length;};
struct Result { uint32_t risk_bits,blocked,error,anchors,windows; };
enum Error : uint32_t { OK=0,BAD_MODEL=1,BAD_WINDOW=2,BAD_ID=3,UNINITIALIZED=4,BAD_CHUNK=5 };
int64_t rne_right(int64_t x,unsigned shift);
int64_t scale_pow2(int64_t x,int shift,int64_t cap);
bool load_model(const uint8_t *blob,uint32_t bytes,Model& model);
uint32_t risk_lookup(const Model& model,int64_t logit);
Result window_reference(const Model&,const uint16_t*,Window,uint32_t);
Result window_streamed(const Model&,const uint16_t*,Window,uint32_t);
Result window_sopa(const Model&,const uint16_t*,Window,uint32_t);
struct Trace {int8_t z[MAX_TOKENS][Z];int32_t p[MAX_TOKENS][2*R],a[MAX_TOKENS][R],b[MAX_TOKENS][R];int64_t g[MAX_TOKENS];uint8_t eligible[MAX_TOKENS];};
Result window_trace(const Model&,const uint16_t*,Window,uint32_t,Trace&);
}
// Continue documents through any number of <=64-window chunks. Previous risk
// carries the exact maximum; zero starts a document. Never truncate an input.
extern "C" void span_guard_top(const uint8_t*,uint32_t,const uint16_t*,
    const sentry_span::Window*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,sentry_span::Result*);
#endif
