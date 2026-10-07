// Goal 4 source only: no physical profile has been selected or synthesized.
// Integer contract: gate_quant.py / static_symmetric_encoder_linear_w4w8a8_fp32_other_v1
#ifndef GATE_LINEAR_ENGINE_HPP
#define GATE_LINEAR_ENGINE_HPP

#include <ap_int.h>

namespace gate {

// A is row-major [rows, inner]; W is row-major [outputs, inner].
// W8: one two's-complement code per byte.
// W4: two two's-complement codes per byte; the earlier code is bits [3:0].
// The integer output is row-major [rows, outputs]. Scales/bias stay outside this
// integer kernel: FP32(acc) * (input_scale * row_weight_scale), then FP32 bias.
inline ap_uint<8> load_weight_code(const ap_uint<8>* weights,
                                  int linear_index, int weight_bits) {
#pragma HLS INLINE
    if (weight_bits == 8) return weights[linear_index];
    const ap_uint<8> packed = weights[linear_index / 2];
    ap_uint<8> code = 0;
    if ((linear_index & 1) == 0) code.range(3, 0) = packed.range(3, 0);
    else code.range(3, 0) = packed.range(7, 4);
    return code;
}

// The 5-bit SIGNED carrier represents either signed-high [-8,7] or unsigned-low
// [0,15] without reinterpreting bit 3 as a sign bit. It does not change W4/W8
// quantization. Physical multiplier/LUT/DSP mapping must be read from synthesis.
inline ap_int<13> atomic_digit_product(ap_int<8> activation, ap_int<5> digit) {
#pragma HLS INLINE
    const ap_int<13> product = activation * digit;
    return product;
}

// ONE template specialization is instantiated by the generated top; layer shapes
// and W4/W8 are runtime controls, not separately resized per-layer datapaths.
// LANES and all tile/buffer dimensions must come from an actual configured
// physical profile. No default lane count, memory port, clock or resource exists.
template<int LANES, int TILE_ROWS, int TILE_OUTPUTS, int TILE_INNER, int MAX_ROWS>
bool linear_engine(const ap_int<8>* activations, const ap_uint<8>* weights,
                   ap_int<32>* output, int rows, int inner, int outputs,
                   int weight_bits) {
#pragma HLS INLINE off
    static_assert(LANES > 0, "LANES must be a selected positive profile value");
    static_assert(TILE_ROWS > 0 && TILE_OUTPUTS > 0 && TILE_INNER > 0,
                  "Tiles must be selected positive profile values");
    static_assert(TILE_INNER % LANES == 0, "Inner tile must divide into lane groups");
    static_assert(MAX_ROWS > 0, "MAX_ROWS must be selected");
    // Only the observed BERT-Mini encoder shapes; all have inner <= 1024.
    const bool shape_ok = (inner == 256 && (outputs == 256 || outputs == 1024))
                       || (inner == 1024 && outputs == 256);
    if (!shape_ok || rows <= 0 || rows > MAX_ROWS ||
        (weight_bits != 4 && weight_bits != 8)) return false;

    ap_int<8> activation_tile[TILE_ROWS][TILE_INNER];
    ap_uint<8> weight_tile[TILE_OUTPUTS][TILE_INNER];
    ap_int<32> partial[TILE_ROWS][TILE_OUTPUTS][LANES];
#pragma HLS ARRAY_PARTITION variable=activation_tile cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tile cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=partial complete dim=3

    // Serialized load/compute/store schedule: no unmeasured DMA overlap/dataflow.
    for (int row_base = 0; row_base < rows; row_base += TILE_ROWS) {
        for (int out_base = 0; out_base < outputs; out_base += TILE_OUTPUTS) {
            for (int r = 0; r < TILE_ROWS; ++r) {
                for (int o = 0; o < TILE_OUTPUTS; ++o) {
                    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                        partial[r][o][lane] = 0;
                    }
                }
            }
            for (int inner_base = 0; inner_base < inner; inner_base += TILE_INNER) {
                for (int r = 0; r < TILE_ROWS; ++r) {
                    for (int k = 0; k < TILE_INNER; ++k) {
                        activation_tile[r][k] =
                            row_base + r < rows && inner_base + k < inner
                            ? activations[(row_base + r) * inner + inner_base + k]
                            : ap_int<8>(0);
                    }
                }
                for (int o = 0; o < TILE_OUTPUTS; ++o) {
                    for (int k = 0; k < TILE_INNER; ++k) {
                        weight_tile[o][k] =
                            out_base + o < outputs && inner_base + k < inner
                            ? load_weight_code(weights, (out_base + o) * inner + inner_base + k,
                                               weight_bits)
                            : ap_uint<8>(0);
                    }
                }
                for (int r = 0; r < TILE_ROWS; ++r) {
                    for (int o = 0; o < TILE_OUTPUTS; ++o) {
                        for (int k_base = 0; k_base < TILE_INNER; k_base += LANES) {
                            // Reuse the same source multiplication for each digit phase.
                            // W8 = 16 * signed(high nibble) + unsigned(low nibble).
                            const int phases = weight_bits == 8 ? 2 : 1;
                            for (int phase = 0; phase < phases; ++phase) {
#pragma HLS PIPELINE II=1
                                for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                                    const int k = k_base + lane;
                                    const ap_uint<8> code = weight_tile[o][k];
                                    ap_int<4> high;
                                    ap_uint<4> low = code.range(3, 0);
                                    if (weight_bits == 8) high.range(3, 0) = code.range(7, 4);
                                    else high.range(3, 0) = code.range(3, 0);
                                    ap_int<5> digit;
                                    if (weight_bits == 8 && phase == 1)
                                        digit = ap_int<5>(ap_uint<5>(low));
                                    else
                                        digit = ap_int<5>(high);
                                    const ap_int<13> product =
                                        atomic_digit_product(activation_tile[r][k], digit);
                                    ap_int<32> contribution = ap_int<32>(product);
                                    // Widen BEFORE shift; no narrow-product overflow.
                                    if (weight_bits == 8 && phase == 0)
                                        contribution = ap_int<32>(contribution << 4);
                                    partial[r][o][lane] += contribution;
                                }
                            }
                        }
                    }
                }
            }
            for (int r = 0; r < TILE_ROWS; ++r) {
                for (int o = 0; o < TILE_OUTPUTS; ++o) {
                    ap_int<32> sum = 0;
                    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                        sum += partial[r][o][lane];
                    }
                    if (row_base + r < rows && out_base + o < outputs)
                        output[(row_base + r) * outputs + out_base + o] = sum;
                }
            }
        }
    }
    return true;
}
} // namespace gate
#endif

