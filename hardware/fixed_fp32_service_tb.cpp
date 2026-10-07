// Small formula/reference checks only. These are not end-to-end BERT validation.
#include <ap_int.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include "fixed_fp32_service.hpp"

extern "C" void gate_fixed_fp32_top(
    int, const float*, const float*, const float*, const ap_int<32>*,
    float*, ap_int<8>*, int, int, int, float, bool, int*);

namespace {
const int depth = 32768; // Match every wrapper pointer's ELEMENT depth.
std::vector<float> x(depth, 0.0f), y(depth, 0.0f), z(depth, 0.0f), out(depth, 0.0f);
std::vector<ap_int<32>> acc(depth, 0);
std::vector<ap_int<8>> codes(depth, 0);
int failures = 0;

struct ErrorReport {
    const char* name;
    double absolute_tolerance;
    double relative_tolerance;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    int count = 0;
    bool check(float actual, long double expected) {
        const double reference = static_cast<double>(expected);
        const double error = std::abs(static_cast<double>(actual) - reference);
        const double relative = std::abs(reference) > 1.0e-30 ? error / std::abs(reference) : 0.0;
        maximum_absolute = std::max(maximum_absolute, error);
        maximum_relative = std::max(maximum_relative, relative);
        ++count;
        if (!std::isfinite(actual) || error > absolute_tolerance + relative_tolerance * std::abs(reference)) {
            std::cerr << "FAIL " << name << " actual=" << actual << " expected=" << reference
                      << " abs_error=" << error << "\n";
            ++failures;
            return false;
        }
        return true;
    }
    void report() const {
        std::cout << "CHECK " << name << " values=" << count
                  << " max_abs=" << maximum_absolute << " max_rel=" << maximum_relative
                  << " atol=" << absolute_tolerance << " rtol=" << relative_tolerance << "\n";
    }
};

void reset_buffers() {
    std::fill(x.begin(), x.end(), 0.0f);
    std::fill(y.begin(), y.end(), 0.0f);
    std::fill(z.begin(), z.end(), 0.0f);
    std::fill(out.begin(), out.end(), -777.0f);
    std::fill(acc.begin(), acc.end(), ap_int<32>(0));
    std::fill(codes.begin(), codes.end(), ap_int<8>(0));
}

bool call(int op, int rows, int width, int outputs=0, float scale=1.0f,
          bool bias=false, bool valid=true) {
    int status = -1;
    gate_fixed_fp32_top(op, x.data(), y.data(), z.data(), acc.data(), out.data(),
                       codes.data(), rows, width, outputs, scale, bias, &status);
    if (status != (valid ? 0 : 1)) {
        std::cerr << "FAIL status op=" << op << " actual=" << status
                  << " expected=" << (valid ? 0 : 1) << "\n";
        ++failures;
        return false;
    }
    return true;
}

void dot_checks() {
    ErrorReport errors{"dot_high_precision_formula", 3.0e-5, 3.0e-5};
    // QK (K64), pooler/classifier (K256), and longest AV reduction (K512).
    // Five outputs exercises the four-lane group plus an output tail.
    const int widths[3] = {64, 256, 512};
    for (int width : widths) {
        reset_buffers();
        const int rows = 2, outputs = 5;
        for (int i=0; i<rows*width; ++i) x[i] = float((i*7)%17 - 8) / 16.0f;
        for (int i=0; i<outputs*width; ++i) y[i] = float((i*11)%23 - 11) / 32.0f;
        for (int o=0; o<outputs; ++o) z[o] = float(o-2) / 8.0f;
        if (call(gate::FP32_DOT, rows, width, outputs, 1.0f, true)) {
            for (int r=0; r<rows; ++r) for (int o=0; o<outputs; ++o) {
                long double expected = z[o];
                for (int k=0; k<width; ++k)
                    expected += static_cast<long double>(x[r*width+k]) * y[o*width+k];
                errors.check(out[r*outputs+o], expected);
            }
        }
    }
    // Maximum panel extent, last cache address, and reuse by a second input row.
    {
        reset_buffers();
        const int rows = 2, outputs = 64, width = 512;
        for (int i=0; i<rows*width; ++i) x[i] = float((i*7)%17 - 8) / 16.0f;
        for (int i=0; i<outputs*width; ++i) y[i] = float((i*11)%23 - 11) / 32.0f;
        for (int o=0; o<outputs; ++o) z[o] = float(o-2) / 8.0f;
        if (call(gate::FP32_DOT, rows, width, outputs, 1.0f, true)) {
            for (int r=0; r<rows; ++r) for (int o=0; o<outputs; ++o) {
                long double expected = z[o];
                for (int k=0; k<width; ++k)
                    expected += static_cast<long double>(x[r*width+k]) * y[o*width+k];
                errors.check(out[r*outputs+o], expected);
            }
        }
    }

    errors.report();
    // Detect multiply-add contraction: separate rounded product gives exactly 0.
    reset_buffers();
    x[0] = -1.0f; y[0] = 1.0f;
    x[1] = 1.0f + std::ldexp(1.0f, -23);
    y[1] = 1.0f - std::ldexp(1.0f, -23);
    if (call(gate::FP32_DOT, 1, 2, 1) && out[0] != 0.0f) {
        std::cerr << "FAIL dot separate-multiply/add cancellation contract\n";
        ++failures;
    }
    std::cout << "CHECK dot_non_fused_cancellation expected=0 actual=" << out[0] << "\n";
}

void quantize_checks() {
    reset_buffers();
    const float values[] = {-1000.0f,-128.5f,-128.0f,-127.5f,-126.5f,-2.5f,-1.5f,
                            -0.5f,-0.0f,0.5f,1.5f,2.5f,125.5f,126.5f,127.0f,127.5f,1000.0f};
    const int expected[] = {-128,-128,-128,-128,-126,-2,-2,0,0,0,2,2,126,126,127,127,127};
    const int count = sizeof(values)/sizeof(values[0]);
    for (int i=0; i<count; ++i) x[i] = values[i] * 0.25f;
    if (call(gate::FP32_QUANTIZE, 1, count, 0, 0.25f)) {
        for (int i=0; i<count; ++i) if (codes[i].to_int() != expected[i]) {
            std::cerr << "FAIL quantize index=" << i << " code=" << codes[i] << "\n";
            ++failures;
        }
    }
    // Division rounding at a non-power-of-two static scale; independent nearbyint reference.
    for (int i=0; i<33; ++i) x[i] = float(i-16) * 0.037f;
    if (call(gate::FP32_QUANTIZE, 1, 33, 0, 0.073f)) {
        for (int i=0; i<33; ++i) {
            volatile float quotient = x[i] / 0.073f;
            const int reference = std::max(-128, std::min(127, int(std::nearbyint(quotient))));
            if (codes[i].to_int() != reference) {
                std::cerr << "FAIL quantize nonbinary-scale index=" << i << "\n";
                ++failures;
            }
        }
    }
    // Adjacent IEEE values at both saturation boundaries for the bit-exact rewrite.
    const float clip_edges[] = {
        std::nextafter(127.0f,0.0f),127.0f,std::nextafter(127.0f,128.0f),
        std::nextafter(-128.0f,0.0f),-128.0f,std::nextafter(-128.0f,-129.0f),
        std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()};
    for (int i=0;i<8;++i)x[i]=clip_edges[i];
    if(call(gate::FP32_QUANTIZE,1,8))
        for(int i=0;i<8;++i) {
            const long double rounded=std::nearbyint(static_cast<long double>(x[i]));
            const int reference=int(std::max(-128.0L,std::min(127.0L,rounded)));
            if(codes[i].to_int()!=reference) {
                std::cerr<<"FAIL quantize adjacent clip boundary index="<<i<<"\n";++failures;
            }
        }
    std::cout << "CHECK quantize exact signed saturation and nearest-even cases\n";
}

void rescale_checks() {
    reset_buffers();
    const std::int32_t integers[] = {std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(), -16777216, 16777216, -127, 0, 127};
    const float converted[] = {-2147483648.0f,2147483648.0f,-16777216.0f,16777216.0f,-127.0f,0.0f,127.0f};
    const int count = sizeof(integers)/sizeof(integers[0]);
    for (int i=0; i<count; ++i) { acc[i]=integers[i]; y[i]=1.0f; }
    ErrorReport cast{"int32_to_fp32",0.0,0.0};
    if (call(gate::FP32_RESCALE_BIAS,1,count))
        for (int i=0; i<count; ++i) cast.check(out[i], converted[i]);
    cast.report();
    // Exact dyadic scale/bias examples, with two rows sharing channel parameters.
    reset_buffers();
    for (int i=0; i<12; ++i) acc[i]=(i-6)*31;
    for (int c=0; c<6; ++c) {y[c]=float(c+1)/32.0f; z[c]=float(c-2)/8.0f;}
    ErrorReport rescale{"rescale_scaleproduct_then_product_then_bias",0.0,0.0};
    if (call(gate::FP32_RESCALE_BIAS,2,6,0,0.25f))
        for (int i=0; i<12; ++i)
            rescale.check(out[i], static_cast<long double>(acc[i].to_int()) * 0.25L * y[i%6] + z[i%6]);
    rescale.report();
    // Natural FFN width: second-half parameters differ, detecting an accidental
    // 512-channel wrap while both rows must reuse all 1024 channel parameters.
    reset_buffers();
    for (int i=0; i<2048; ++i) acc[i]=(i%37-18)*17;
    for (int c=0; c<1024; ++c) {
        y[c]=float(c+1)/2048.0f;
        z[c]=float(c-511)/1024.0f;
    }
    ErrorReport ffn{"rescale_ffn_2rows_1024channels",0.0,0.0};
    if (call(gate::FP32_RESCALE_BIAS,2,1024,0,0.25f))
        for (int i=0; i<2048; ++i)
            ffn.check(out[i], static_cast<long double>(acc[i].to_int()) * 0.25L * y[i%1024] + z[i%1024]);
    ffn.report();
}

void addition_checks() {
    reset_buffers();
    for (int i=0; i<19; ++i) {x[i]=float(i-9)/8.0f; y[i]=float(i%5-2)/4.0f; z[i]=float(i%3-1)/2.0f;}
    ErrorReport embed{"embedding_word_type_then_position",0.0,0.0};
    if (call(gate::FP32_EMBED_ADD,1,19))
        for (int i=0; i<19; ++i) embed.check(out[i], static_cast<long double>(x[i])+y[i]+z[i]);
    embed.report();
    ErrorReport residual{"residual_add",0.0,0.0};
    if (call(gate::FP32_RESIDUAL_ADD,1,19))
        for (int i=0; i<19; ++i) residual.check(out[i], static_cast<long double>(x[i])+y[i]);
    residual.report();
    // The installed BERT ordering is (word+type)+position, not word+(type+position).
    x[0]=16777216.0f; y[0]=-16777216.0f; z[0]=1.0f;
    if (call(gate::FP32_EMBED_ADD,1,1) && out[0] != 1.0f) {
        std::cerr << "FAIL embedding addition order\n"; ++failures;
    }
}

void layer_norm_checks() {
    reset_buffers();
    const int width=256, rows=3;
    for (int c=0; c<width; ++c) {
        x[c]=1.25f; // Zero variance: output must be beta.
        x[width+c]=float(c%13-6)/8.0f;
        x[2*width+c]=128.0f+float(c%9-4)/4.0f; // Common offset; centered variance.
        y[c]=1.0f+float(c%5-2)/16.0f;
        z[c]=float(c%7-3)/32.0f;
    }
    ErrorReport errors{"layer_norm_centered_variance_eps1e-12",2.0e-4,2.0e-4};
    if (call(gate::FP32_LAYER_NORM,rows,width)) {
        for (int r=0; r<rows; ++r) {
            long double mean=0.0L, variance=0.0L;
            for (int c=0; c<width; ++c) mean+=x[r*width+c];
            mean/=width;
            for (int c=0; c<width; ++c) {
                const long double centered=static_cast<long double>(x[r*width+c])-mean;
                variance+=centered*centered;
            }
            variance/=width;
            for (int c=0; c<width; ++c)
                errors.check(out[r*width+c],
                    ((static_cast<long double>(x[r*width+c])-mean)/std::sqrt(variance+1.0e-12L))*y[c]+z[c]);
        }
    }
    errors.report();
}

void softmax_checks() {
    ErrorReport errors{"stable_softmax_scaling_mask",3.0e-6,3.0e-5};
    const int widths[3]={2,17,512};
    for (int width:widths) {
        reset_buffers();
        const float scale=width==2 ? 1.0f : 8.0f;
        for (int i=0; i<width; ++i) {
            x[i]=1000.0f+float(i%17-8);
            y[i]=i%11==0 && i!=0 ? -std::numeric_limits<float>::max() : 0.0f;
        }
        if (call(gate::FP32_SOFTMAX,1,width,0,scale)) {
            std::vector<long double> scores(width);
            long double maximum=-std::numeric_limits<long double>::max(), sum=0.0L;
            for (int i=0; i<width; ++i) {
                scores[i]=static_cast<long double>(x[i])/scale+y[i];
                maximum=std::max(maximum,scores[i]);
            }
            for (int i=0; i<width; ++i) {scores[i]=std::exp(scores[i]-maximum); sum+=scores[i];}
            double actual_sum=0.0;
            for (int i=0; i<width; ++i) {errors.check(out[i],scores[i]/sum);actual_sum+=out[i];}
            if (std::abs(actual_sum-1.0)>2.0e-5) {
                std::cerr << "FAIL softmax normalization sum=" << actual_sum << "\n"; ++failures;
            }
        }
    }
    errors.report();
}

void nonlinear_checks() {
    reset_buffers();
    const float values[]={-10.0f,-3.0f,-1.0f,-0.125f,-0.0f,0.0f,0.125f,1.0f,3.0f,10.0f};
    const int count=sizeof(values)/sizeof(values[0]);
    for(int i=0;i<count;++i)x[i]=values[i];
    ErrorReport gelu{"erf_GELU_formula",3.0e-6,3.0e-5};
    if(call(gate::FP32_GELU,1,count))
        for(int i=0;i<count;++i) {
            const long double value=x[i];
            gelu.check(out[i],0.5L*value*(1.0L+std::erf(value/std::sqrt(2.0L))));
        }
    gelu.report();
    ErrorReport tanh{"tanh_formula",3.0e-6,3.0e-5};
    if(call(gate::FP32_TANH,1,count))
        for(int i=0;i<count;++i)tanh.check(out[i],std::tanh(static_cast<long double>(x[i])));
    tanh.report();
}

void invalid_checks() {
    reset_buffers();
    call(-1,1,8,0,1.0f,false,false);
    call(gate::FP32_DOT,1,64,65,1.0f,false,false);
    call(gate::FP32_DOT,512,512,4,1.0f,false,false); // span >32768.
    call(gate::FP32_QUANTIZE,1,8,0,0.0f,false,false);
    call(gate::FP32_QUANTIZE,1,8,0,-1.0f,false,false);
    call(gate::FP32_QUANTIZE,1,8,0,std::numeric_limits<float>::quiet_NaN(),false,false);
    call(gate::FP32_LAYER_NORM,1,255,0,1.0f,false,false);
    call(gate::FP32_SOFTMAX,1,513,0,1.0f,false,false);
    call(gate::FP32_TANH,0,8,0,1.0f,false,false);
    call(gate::FP32_GELU,1,1025,0,1.0f,false,false);
    if(out[0]!=-777.0f) {std::cerr<<"FAIL invalid control wrote output\n";++failures;}
    x[0]=std::numeric_limits<float>::quiet_NaN();
    call(gate::FP32_GELU,1,1,0,1.0f,false,false);
    std::cout<<"CHECK invalid op/dimensions/scales/nonfinite input\n";
    reset_buffers();
    x[0]=std::numeric_limits<float>::max(); y[0]=2.0f;
    call(gate::FP32_DOT,1,1,1,1.0f,false,false);
    reset_buffers();
    acc[0]=1; y[0]=2.0f;
    call(gate::FP32_RESCALE_BIAS,1,1,0,std::numeric_limits<float>::max(),false,false);
    reset_buffers();
    x[0]=std::numeric_limits<float>::max(); y[0]=x[0];
    call(gate::FP32_RESIDUAL_ADD,1,1,0,1.0f,false,false);
    call(gate::FP32_EMBED_ADD,1,1,0,1.0f,false,false);
    reset_buffers();
    for(int i=0;i<256;++i) {x[i]=std::numeric_limits<float>::max(); y[i]=1.0f;}
    call(gate::FP32_LAYER_NORM,1,256,0,1.0f,false,false);
    // Finite mean but squared centered values overflow; do not return zero vectors.
    for(int i=0;i<256;++i)x[i]=(i%2==0 ? 1.0f : -1.0f)*1.0e30f;
    call(gate::FP32_LAYER_NORM,1,256,0,1.0f,false,false);
    std::cout<<"CHECK arithmetic overflow is invalid, not a successful nonfinite output\n";
}
} // namespace

int main() {
    if(std::fesetround(FE_TONEAREST)!=0) {
        std::cerr<<"Cannot establish nearest-even host reference rounding.\n";return 2;
    }
    dot_checks();
    quantize_checks();
    rescale_checks();
    addition_checks();
    layer_norm_checks();
    softmax_checks();
    nonlinear_checks();
    invalid_checks();
    if(failures) {
        std::cerr<<"FAIL fixed_fp32 service checks="<<failures<<"\n";return 1;
    }
    std::cout<<"PASS fixed_fp32 service\n";
    return 0;
}
