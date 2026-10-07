// Goal 4 common-baseline weight-reuse pilot; source is not timing evidence.
// Integer arithmetic remains the contract in linear_engine.hpp / gate_quant.py.
#ifndef GATE_LINEAR_ENGINE_CACHED_HPP
#define GATE_LINEAR_ENGINE_CACHED_HPP

#include "linear_engine.hpp"

namespace gate {

// The largest observed matrix is 256*1024 or 1024*256, not 1024*1024.
// A flat byte-code cache occupies 256 KiB logically for either precision.
// W4 preload reads each packed byte once and expands its two codes on chip.
// Per valid call, external weight reads are outputs*inner bytes for W8 and
// outputs*inner/2 bytes for W4. These source counts are not measured AXI beats.
// Activations retain the original per-output-tile reload schedule. No overlap,
// achieved II, clock, physical RAM use or speedup is asserted by this source.
template<int LANES, int TILE_ROWS, int TILE_OUTPUTS, int TILE_INNER, int MAX_ROWS>
bool linear_engine_cached(const ap_int<8>* activations,
                          const ap_uint<8>* weights, ap_int<32>* output,
                          int rows, int inner, int outputs, int weight_bits) {
#pragma HLS INLINE off
    static_assert(LANES > 0, "LANES must be a selected positive profile value");
    static_assert(TILE_ROWS == 4 && TILE_OUTPUTS == 16 && TILE_INNER == 64,
                  "This pilot preserves the characterized 4/16/64 tiles");
    static_assert(TILE_INNER % LANES == 0,
                  "Inner tile must divide into lane groups");
    static_assert(MAX_ROWS == 256,
                  "This pilot preserves the characterized maximum 256 rows");

    const int MAX_WEIGHT_CODES = 262144;
    const bool shape_ok = (inner == 256 && (outputs == 256 || outputs == 1024))
                       || (inner == 1024 && outputs == 256);
    // Reject unsupported dimensions before calculating any cache index or load.
    if (!shape_ok || rows <= 0 || rows > MAX_ROWS ||
        (weight_bits != 4 && weight_bits != 8)) return false;
    const int weight_codes = outputs * inner;
    if (weight_codes <= 0 || weight_codes > MAX_WEIGHT_CODES ||
        (weight_codes & 1) != 0) return false;

    ap_uint<8> weight_cache[MAX_WEIGHT_CODES];
    ap_int<8> activation_tile[TILE_ROWS][TILE_INNER];
    ap_int<32> partial[TILE_ROWS][TILE_OUTPUTS][LANES];
#pragma HLS ARRAY_PARTITION variable=weight_cache cyclic factor=LANES dim=1
#pragma HLS BIND_STORAGE variable=weight_cache type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=activation_tile cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=partial complete dim=3

    // Preload once per call. There is no persistent-weight/cache-hit assumption.
    if (weight_bits == 8) {
        for (int index = 0; index < weight_codes; ++index) {
#pragma HLS PIPELINE II=1
            weight_cache[index] = weights[index];
        }
    } else {
        for (int packed_index = 0; packed_index < weight_codes / 2; ++packed_index) {
#pragma HLS PIPELINE II=1
            const ap_uint<8> packed = weights[packed_index];
            ap_uint<8> first_code = 0;
            ap_uint<8> second_code = 0;
            first_code.range(3, 0) = packed.range(3, 0);
            second_code.range(3, 0) = packed.range(7, 4);
            weight_cache[2 * packed_index] = first_code;
            weight_cache[2 * packed_index + 1] = second_code;
        }
    }

    // Keep activation loads, digit order, integer partials and output reduction
    // unchanged. No serialized weight-tile copy follows the one-time preload.
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
                for (int r = 0; r < TILE_ROWS; ++r) {
                    for (int o = 0; o < TILE_OUTPUTS; ++o) {
                        for (int k_base = 0; k_base < TILE_INNER; k_base += LANES) {
                            const int phases = weight_bits == 8 ? 2 : 1;
                            for (int phase = 0; phase < phases; ++phase) {
#pragma HLS PIPELINE II=1
                                for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                                    const int k = k_base + lane;
                                    // K and tile bases are multiples of LANES, so
                                    // adjacent lane codes use distinct cyclic banks.
                                    const int cache_index =
                                        (out_base + o) * inner + inner_base + k;
                                    const ap_uint<8> code = weight_cache[cache_index];
                                    ap_int<4> high;
                                    ap_uint<4> low = code.range(3, 0);
                                    if (weight_bits == 8)
                                        high.range(3, 0) = code.range(7, 4);
                                    else
                                        high.range(3, 0) = code.range(3, 0);
                                    ap_int<5> digit;
                                    if (weight_bits == 8 && phase == 1)
                                        digit = ap_int<5>(ap_uint<5>(low));
                                    else
                                        digit = ap_int<5>(high);
                                    const ap_int<13> product =
                                        atomic_digit_product(activation_tile[r][k], digit);
                                    ap_int<32> contribution = ap_int<32>(product);
                                    // Widen before shifting, exactly as in the
                                    // original engine's signed-high reconstruction.
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
