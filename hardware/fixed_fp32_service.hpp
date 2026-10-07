// Goal 4 bounded shared FP32 service. Timing, sharing and II require HLS reports.
// Formula agreement is not end-to-end agreement with PyTorch's reduction kernels.
#ifndef GATE_FIXED_FP32_SERVICE_HPP
#define GATE_FIXED_FP32_SERVICE_HPP
#define GATE_FIXED_FP32_SCHEDULE_REVISION "panel_cache_batched_attention_v1"

#include <ap_int.h>
#include <hls_math.h>
#include <utils/x_hls_utils.h>
#include <cfloat>

namespace gate {

enum FixedFp32Op {
    FP32_DOT = 0,
    FP32_QUANTIZE = 1,
    FP32_RESCALE_BIAS = 2,
    FP32_EMBED_ADD = 3,
    FP32_RESIDUAL_ADD = 4,
    FP32_LAYER_NORM = 5,
    FP32_SOFTMAX = 6,
    FP32_GELU = 7,
    FP32_TANH = 8
};

static const int FP32_MAX_ELEMENTS = 32768;
static const int FP32_MAX_WIDTH = 512; // Buffered DOT/SOFTMAX rows.
static const int FP32_MAX_STREAM_WIDTH = 1024; // FFN vectors; no row buffering.
static const int FP32_DOT_LANES = 4;

// Separate non-inlined operations and explicit rounded stores prohibit implicit
// multiply-add contraction in this source. The driver must also compile with
// -ffp-contract=off and unsafe_math_optimizations=false; inspect generated RTL.
static float fp32_add(float a, float b) {
#pragma HLS INLINE off
    float value = a + b;
#pragma HLS BIND_OP variable=value op=fadd
    volatile float rounded = value;
    return rounded;
}

static float fp32_mul(float a, float b) {
#pragma HLS INLINE off
    float value = a * b;
#pragma HLS BIND_OP variable=value op=fmul
    volatile float rounded = value;
    return rounded;
}

static float fp32_div(float a, float b) {
#pragma HLS INLINE off
    float value = a / b;
#pragma HLS BIND_OP variable=value op=fdiv
    volatile float rounded = value;
    return rounded;
}

static float fp32_sqrt(float x) {
#pragma HLS INLINE off
    return hls::sqrtf(x);
}

static float fp32_exp(float x) {
#pragma HLS INLINE off
    return hls::expf(x);
}

static float fp32_erf(float x) {
#pragma HLS INLINE off
    return hls::erff(x);
}

static float fp32_tanh(float x) {
#pragma HLS INLINE off
    return hls::tanhf(x);
}

inline bool fp32_finite(float x) {
#pragma HLS INLINE
    // IEEE exponent classification is exact, including subnormals and signed zero.
    const ::fp_struct<float> bits(x);
    return bits.exp != 255;
}

inline ap_int<8> fp32_round_clip_a8(float scaled) {
#pragma HLS INLINE
    // Clip before the integer conversion so very large finite inputs, including
    // overflow to infinity during division, cannot overflow a C++ int cast.
    // On either sign, positive IEEE magnitude bits are monotonically ordered.
    // 127.0f = 0x42fe0000 and |(-128.0f)| = 0x43000000. The caller rejects NaN.
    // This preserves comparisons for every finite input and both infinities,
    // while avoiding the measured serial floating-compare/mux timing path.
    const ::fp_struct<float> bits(scaled);
    const ap_uint<32> raw = bits.data();
    const ap_uint<31> magnitude = raw.range(30, 0);
    if (!bits.sign && magnitude >= 0x42fe0000U) return ap_int<8>(127);
    if (bits.sign && magnitude >= 0x43000000U) return ap_int<8>(-128);
    const float lower = hls::floorf(scaled);
    const int integer = static_cast<int>(lower); // Now bounded [-128,126].
    const float fraction = scaled - lower;
    const bool increment = fraction > 0.5f
                        || (fraction == 0.5f && integer % 2 != 0);
    return ap_int<8>(integer + (increment ? 1 : 0));
}

// All AXI pointer depths are 32768 ELEMENTS in the wrapper, not bytes.
// DOT: X[rows,width], Y[outputs,width], optional Z[outputs] bias, output[rows,outputs].
// Other ops: X/output/codes[rows,width]; outputs and dot_bias are ignored.
// RESCALE: accumulators[rows,width], Y[width] weight scales, Z[width] bias;
//          output=float(accumulator)*(scale*Y[channel])+Z[channel].
// EMBED: (X[word]+Y[token_type])+Z[position], matching installed BERT ordering.
// LN: width=256, Y[width]=gamma, Z[width]=beta, biased centered variance, eps=1e-12.
// SOFTMAX: each row normalizes X/scale + Y[rows,width] additive mask.
//          Attention uses scale=8; classifier uses scale=1 and zero mask.
// Inputs are finite model values. Positive finite scales are required where used.
// false means output is unusable (a data error can occur after earlier writes).
static bool fixed_fp32_service(
        int op, const float* x, const float* y, const float* z,
        const ap_int<32>* accumulators, float* output, ap_int<8>* codes,
        int rows, int width, int outputs, float scale, bool dot_bias) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=fp32_add limit=4
#pragma HLS ALLOCATION function instances=fp32_mul limit=4
#pragma HLS ALLOCATION function instances=fp32_div limit=1
#pragma HLS ALLOCATION function instances=fp32_sqrt limit=1
#pragma HLS ALLOCATION function instances=fp32_exp limit=1
#pragma HLS ALLOCATION function instances=fp32_erf limit=1
#pragma HLS ALLOCATION function instances=fp32_tanh limit=1

    if (op < FP32_DOT || op > FP32_TANH || rows <= 0 || rows > 512
            || width <= 0 || width > FP32_MAX_STREAM_WIDTH
            || rows * width > FP32_MAX_ELEMENTS) return false;
    if ((op == FP32_QUANTIZE || op == FP32_RESCALE_BIAS || op == FP32_SOFTMAX)
            && (!fp32_finite(scale) || scale <= 0.0f)) return false;
    if ((op == FP32_DOT || op == FP32_SOFTMAX) && width > FP32_MAX_WIDTH) return false;
    if (op == FP32_LAYER_NORM && width != 256) return false;
    if (op == FP32_DOT && (outputs <= 0 || outputs > 64
            || outputs * width > FP32_MAX_ELEMENTS
            || rows * outputs > FP32_MAX_ELEMENTS)) return false;

    float row[FP32_MAX_WIDTH];
    // One full N<=64, K<=512 panel is reused for all M input rows in this call.
    float weights[64][FP32_MAX_WIDTH];
    float partial[FP32_DOT_LANES];
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4 dim=1
#pragma HLS BIND_STORAGE variable=weights type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1

    if (op == FP32_DOT) {
        bool panel_valid = true;
        dot_panel_outputs:
        for (int o = 0; o < outputs; ++o) {
            dot_load_panel:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const float value = y[o * width + k];
                panel_valid = panel_valid & fp32_finite(value);
                weights[o][k] = value;
            }
        }
        if (!panel_valid) return false;
        // Four outputs proceed in parallel; each sums k=0..width-1 in order.
        for (int r = 0; r < rows; ++r) {
            bool row_valid = true;
            dot_load_row:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const float value = x[r * width + k];
                row_valid = row_valid & fp32_finite(value);
                row[k] = value;
            }
            if (!row_valid) return false;
            for (int base = 0; base < outputs; base += FP32_DOT_LANES) {
                for (int lane = 0; lane < FP32_DOT_LANES; ++lane) {
#pragma HLS UNROLL
                    partial[lane] = 0.0f;
                }
                dot_reduce:
                for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                    for (int lane = 0; lane < FP32_DOT_LANES; ++lane) {
#pragma HLS UNROLL
                        // Never read an uninitialized tail row. Valid lanes keep
                        // the exact same multiply and ascending add sequence.
                        const float weight = base + lane < outputs ? weights[base + lane][k] : 0.0f;
                        const float product = fp32_mul(row[k], weight);
                        partial[lane] = fp32_add(partial[lane], product);
                    }
                }
                for (int lane = 0; lane < FP32_DOT_LANES; ++lane) {
#pragma HLS UNROLL
                    if (base + lane < outputs) {
                        float result = partial[lane];
                        if (dot_bias) {
                            const float bias = z[base + lane];
                            if (!fp32_finite(bias)) return false;
                            result = fp32_add(result, bias);
                        }
                        if (!fp32_finite(result)) return false;
                        output[r * outputs + base + lane] = result;
                    }
                }
            }
        }
        return true;
    }


    if (op == FP32_LAYER_NORM) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            bool inputs_valid = true;
            ln_load_sum:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const float input = x[r * width + k];
                const bool input_ok = fp32_finite(input);
                const float value = input_ok ? input : 0.0f;
                inputs_valid = inputs_valid & input_ok;
                row[k] = value;
                sum = fp32_add(sum, value);
            }
            if (!inputs_valid || !fp32_finite(sum)) return false;
            const float mean = fp32_div(sum, 256.0f);
            float centered_square_sum = 0.0f;
            ln_variance:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const float difference = fp32_add(row[k], -mean);
                centered_square_sum = fp32_add(centered_square_sum,
                                                fp32_mul(difference, difference));
            }
            // Centered two-pass variance avoids E[x*x]-E[x]^2 cancellation.
            if (!fp32_finite(centered_square_sum)) return false;
            const float variance = fp32_div(centered_square_sum, 256.0f);
            const float denominator = fp32_sqrt(fp32_add(variance, 1.0e-12f));
            if (!fp32_finite(denominator) || denominator <= 0.0f) return false;
            bool parameters_valid = true, results_valid = true;
            ln_normalize:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const float gamma = y[k], beta = z[k];
                const bool gamma_ok = fp32_finite(gamma), beta_ok = fp32_finite(beta);
                const float centered = fp32_add(row[k], -mean);
                const float normalized = fp32_div(centered, denominator);
                const float result = fp32_add(
                    fp32_mul(normalized, gamma_ok ? gamma : 0.0f), beta_ok ? beta : 0.0f);
                parameters_valid = parameters_valid & gamma_ok & beta_ok;
                results_valid = results_valid & fp32_finite(result);
                output[r * width + k] = result;
            }
            if (!parameters_valid || !results_valid) return false;
        }
        return true;
    }

    if (op == FP32_SOFTMAX) {
        for (int r = 0; r < rows; ++r) {
            float maximum = -FLT_MAX;
            bool inputs_valid = true, scores_valid = true;
            softmax_scale_max:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                const int index = r * width + k;
                const float input = x[index], mask = y[index];
                const bool input_ok = fp32_finite(input), mask_ok = fp32_finite(mask);
                const float score = fp32_add(
                    fp32_div(input_ok ? input : 0.0f, scale), mask_ok ? mask : 0.0f);
                const bool score_ok = fp32_finite(score);
                const float safe_score = score_ok ? score : 0.0f;
                inputs_valid = inputs_valid & input_ok & mask_ok;
                scores_valid = scores_valid & score_ok;
                row[k] = safe_score;
                if (safe_score > maximum) maximum = safe_score;
            }
            if (!inputs_valid || !scores_valid) return false;
            float sum = 0.0f;
            softmax_exp_sum:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                row[k] = fp32_exp(fp32_add(row[k], -maximum));
                sum = fp32_add(sum, row[k]);
            }
            if (!(sum > 0.0f) || !fp32_finite(sum)) return false;
            softmax_normalize:
            for (int k = 0; k < width; ++k) {
#pragma HLS PIPELINE
                output[r * width + k] = fp32_div(row[k], sum);
            }
        }
        return true;
    }


    // Each mode keeps validity reductions separate from its data pipeline.
    // A failed call may have written outputs; callers must ignore them.
    if (op == FP32_QUANTIZE) {
        bool inputs_valid = true, quotients_valid = true;
        quantize_elements:
        for (int index = 0; index < rows * width; ++index) {
#pragma HLS PIPELINE
            const float value = x[index];
            const bool input_ok = fp32_finite(value);
            const float scaled = fp32_div(input_ok ? value : 0.0f, scale);
            const bool quotient_ok = scaled == scaled;
            inputs_valid = inputs_valid & input_ok;
            quotients_valid = quotients_valid & quotient_ok;
            // Preserve +/-Inf saturation; only NaN is replaced before floor/cast.
            codes[index] = fp32_round_clip_a8(quotient_ok ? scaled : 0.0f);
        }
        return inputs_valid & quotients_valid;
    }

    if (op == FP32_RESCALE_BIAS) {
        bool parameters_valid = true, factors_valid = true, results_valid = true;
        rescale_rows:
        for (int r = 0; r < rows; ++r) {
            rescale_channels:
            for (int channel = 0; channel < width; ++channel) {
#pragma HLS PIPELINE
                const int index = r * width + channel;
                const float weight_scale = y[channel], bias = z[channel];
                const bool weight_ok = fp32_finite(weight_scale) && weight_scale > 0.0f;
                const bool bias_ok = fp32_finite(bias);
                const float combined_scale = fp32_mul(scale, weight_ok ? weight_scale : 1.0f);
                const bool factor_ok = fp32_finite(combined_scale) && combined_scale > 0.0f;
                const float integer_as_float = static_cast<float>(accumulators[index].to_int());
                const float result = fp32_add(
                    fp32_mul(integer_as_float, factor_ok ? combined_scale : 1.0f),
                    bias_ok ? bias : 0.0f);
                parameters_valid = parameters_valid & weight_ok & bias_ok;
                factors_valid = factors_valid & factor_ok;
                results_valid = results_valid & fp32_finite(result);
                output[index] = result;
            }
        }
        return parameters_valid & factors_valid & results_valid;
    }

    if (op == FP32_EMBED_ADD) {
        bool inputs_valid = true, results_valid = true;
        embedding_elements:
        for (int index = 0; index < rows * width; ++index) {
#pragma HLS PIPELINE
            const float value = x[index], token_type = y[index], position = z[index];
            const bool word_ok = fp32_finite(value), type_ok = fp32_finite(token_type);
            const bool position_ok = fp32_finite(position);
            const float result = fp32_add(
                fp32_add(word_ok ? value : 0.0f, type_ok ? token_type : 0.0f),
                position_ok ? position : 0.0f);
            inputs_valid = inputs_valid & word_ok & type_ok & position_ok;
            results_valid = results_valid & fp32_finite(result);
            output[index] = result;
        }
        return inputs_valid & results_valid;
    }

    if (op == FP32_RESIDUAL_ADD) {
        bool inputs_valid = true, results_valid = true;
        residual_elements:
        for (int index = 0; index < rows * width; ++index) {
#pragma HLS PIPELINE
            const float value = x[index], residual = y[index];
            const bool value_ok = fp32_finite(value), residual_ok = fp32_finite(residual);
            const float result = fp32_add(value_ok ? value : 0.0f, residual_ok ? residual : 0.0f);
            inputs_valid = inputs_valid & value_ok & residual_ok;
            results_valid = results_valid & fp32_finite(result);
            output[index] = result;
        }
        return inputs_valid & results_valid;
    }

    if (op == FP32_GELU) {
        bool inputs_valid = true, results_valid = true;
        gelu_elements:
        for (int index = 0; index < rows * width; ++index) {
#pragma HLS PIPELINE
            const float input = x[index];
            const bool input_ok = fp32_finite(input);
            const float value = input_ok ? input : 0.0f;
            // Same erf GELU and FP32 grouping for every valid input.
            const float argument = fp32_div(value, 1.4142135623730951f);
            const float half_value = fp32_mul(value, 0.5f);
            const float result = fp32_mul(half_value, fp32_add(1.0f, fp32_erf(argument)));
            inputs_valid = inputs_valid & input_ok;
            results_valid = results_valid & fp32_finite(result);
            output[index] = result;
        }
        return inputs_valid & results_valid;
    }

    if (op == FP32_TANH) {
        bool inputs_valid = true, results_valid = true;
        tanh_elements:
        for (int index = 0; index < rows * width; ++index) {
#pragma HLS PIPELINE
            const float input = x[index];
            const bool input_ok = fp32_finite(input);
            const float result = fp32_tanh(input_ok ? input : 0.0f);
            inputs_valid = inputs_valid & input_ok;
            results_valid = results_valid & fp32_finite(result);
            output[index] = result;
        }
        return inputs_valid & results_valid;
    }
    return false; // Exhaustive valid-mode dispatch above.
}

} // namespace gate
#endif
