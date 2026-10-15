#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
void int8_tile(adf::input_buffer<int8>&,adf::input_buffer<int8>&,adf::output_buffer<int32>&);
void bf16_tile(adf::input_buffer<bfloat16>&,adf::input_buffer<bfloat16>&,adf::output_buffer<float>&);
