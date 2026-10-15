#include "tiles.hpp"
// Standard native mmul micro-tiles, adapted from the installed AMD 2025.2
// aie-ml_sys_design example's use of aie::mmul. No claimed new arithmetic.
void int8_tile(adf::input_buffer<int8>& a,adf::input_buffer<int8>& b,adf::output_buffer<int32>& c) {
    auto pa=aie::begin_vector<64>(a);auto pb=aie::begin_vector<128>(b);
    for(int row=0;row<8;++row) for(int col=0;col<4;++col) {
        aie::mmul<4,16,8,int8,int8,acc32> acc;
        acc.mul(*pa++,*pb++);
        for(int k=1;k<16;++k) chess_prepare_for_pipelining {acc.mac(*pa++,*pb++);}
        // Output native tile order [row/4,col/8,row%4,col%8].
        aie::store_v(c.data()+(row*4+col)*32,acc.to_vector<int32>());
        if(col<3)pa-=16;else pb-=64;
    }
}
void bf16_tile(adf::input_buffer<bfloat16>& a,adf::input_buffer<bfloat16>& b,adf::output_buffer<float>& c) {
    auto pa=aie::begin_vector<32>(a);auto pb=aie::begin_vector<64>(b);
    for(int row=0;row<8;++row)for(int col=0;col<4;++col) {
        aie::mmul<4,8,8,bfloat16,bfloat16,accfloat> acc;
        acc.mul(*pa++,*pb++);
        for(int k=1;k<8;++k) chess_prepare_for_pipelining {acc.mac(*pa++,*pb++);}
        aie::store_v(c.data()+(row*4+col)*32,acc.to_vector<float>());
        if(col<3)pa-=8;else pb-=32;
    }
}
