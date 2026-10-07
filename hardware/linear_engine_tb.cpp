// Default mode checks all six signed cases. Point mode records two identical transactions.
#include <ap_int.h>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "selected_profile.hpp"

extern "C" void gate_linear_top(const ap_int<8>*, const ap_uint<8>*, ap_int<32>*,
                                int, int, int, int, int*);

static bool positive_integer(const char* text, int& result) {
    if (!text || text[0] < '0' || text[0] > '9') return false;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value < 1 || value > INT_MAX) return false;
    result = static_cast<int>(value);
    return true;
}

static int run_case(int rows, int inner, int outputs, int bits, int repetitions,
                    bool check_invalid_shape) {
    const int activation_values[5] = {-128,-1,0,1,127};
    const int w4_values[5] = {-8,-1,0,1,7};
    const int w8_values[9] = {-128,-127,-17,-16,-1,0,15,16,127};
    // The co-simulation wrapper copies the full declared m_axi pointer depth.
    // Match depth=262144 on activations, weights and output; unused entries are
    // zero-padded while initialization/reference checks retain logical dimensions.
    const int interface_depth = 262144;
    std::vector<ap_int<8>> a(interface_depth, 0);
    std::vector<ap_uint<8>> packed(interface_depth, 0);
    std::vector<int> w(outputs * inner);
    for (int i = 0; i < rows * inner; ++i) a[i] = activation_values[i % 5];
    for (int i = 0; i < outputs * inner; ++i) {
        w[i] = bits == 4 ? w4_values[(i * 3 + 1) % 5] : w8_values[(i * 5 + 2) % 9];
        if (bits == 8) packed[i] = ap_uint<8>(w[i] & 255);
        else {
            ap_uint<8> byte = packed[i / 2];
            const ap_uint<4> nibble = ap_uint<4>(w[i] & 15);
            if ((i & 1) == 0) byte.range(3,0) = nibble;
            else byte.range(7,4) = nibble;
            packed[i / 2] = byte;
        }
    }
    for (int repetition = 1; repetition <= repetitions; ++repetition) {
        std::vector<ap_int<32>> actual(interface_depth, 0);
        int status = -1;
        // The same arrays and runtime controls are supplied to every transaction.
        gate_linear_top(a.data(), packed.data(), actual.data(), rows, inner, outputs, bits, &status);
        if (status != 0) {
            std::cerr << "Valid case returned status=" << status
                      << " repetition=" << repetition << "\n";
            return 1;
        }
        for (int r = 0; r < rows; ++r) {
            for (int o = 0; o < outputs; ++o) {
                std::int64_t expected = 0;
                for (int k = 0; k < inner; ++k)
                    expected += std::int64_t(a[r * inner + k].to_int()) * w[o * inner + k];
                if (expected != actual[r * outputs + o].to_int64()) {
                    std::cerr << "Integer mismatch at " << bits << "," << inner << ","
                              << outputs << "," << r << "," << o
                              << " repetition=" << repetition << "\n";
                    return 2;
                }
            }
        }
        std::cout << "PASS case rows=" << rows << " inner=" << inner
                  << " outputs=" << outputs << " bits=" << bits
                  << " repetition=" << repetition << " repetitions=" << repetitions << "\n";
        if (check_invalid_shape) {
            // Reuse ample valid buffers; inner=128 is outside the fixed BERT shapes.
            status = -1;
            gate_linear_top(a.data(), packed.data(), actual.data(), rows, 128, outputs, bits, &status);
            if (status != 1) {
                std::cerr << "Invalid shape was not rejected with status 1.\n";
                return 3;
            }
            std::cout << "PASS invalid inner=128 status=1\n";
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 1) {
        int values[5];
        if (argc != 6) {
            std::cerr << "Usage: testbench [rows inner outputs bits repetitions]\n";
            return 64;
        }
        for (int i = 0; i < 5; ++i) {
            if (!positive_integer(argv[i + 1], values[i])) {
                std::cerr << "Point arguments must be positive decimal integers.\n";
                return 64;
            }
        }
        const int rows = values[0], inner = values[1], outputs = values[2];
        const int bits = values[3], repetitions = values[4];
        const bool shape_ok = (inner == 256 && (outputs == 256 || outputs == 1024))
                           || (inner == 1024 && outputs == 256);
        if (rows > GATE_MAX_ROWS || !shape_ok || (bits != 4 && bits != 8)
            || repetitions != 2) {
            std::cerr << "Point requires rows<=GATE_MAX_ROWS, a supported BERT shape, "
                         "bits=4 or 8, and repetitions=2.\n";
            return 64;
        }
        return run_case(rows, inner, outputs, bits, repetitions, false);
    }

    const int shapes[3][2] = {{256,256}, {256,1024}, {1024,256}}; // inner, outputs
    // Exercise a complete row tile and a second, partial tile when supported.
    const int rows = GATE_TILE_ROWS < GATE_MAX_ROWS ? GATE_TILE_ROWS + 1 : GATE_MAX_ROWS;
    for (int shape = 0; shape < 3; ++shape) {
        for (int bits = 4; bits <= 8; bits += 4) {
            const int result = run_case(rows, shapes[shape][0], shapes[shape][1], bits, 1,
                                        shape == 0 && bits == 4);
            if (result != 0) return result;
        }
    }
    std::cout << "Representative signed W4/W8 matrices agree with int64 reference.\n";
    return 0;
}
