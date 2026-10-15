#define BERT_HLS
#include "model.hpp"
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
using Axis=ap_axiu<128,0,0,0>;
using Stream=hls::stream<Axis>;
struct AieTiles {
    Stream &ia,&ib,&ic,&fa,&fb,&fc;
    void i8(const int8_t* a,const int8_t* b,int32_t* c) {
        // Reuse each B panel over eight four-row native mmul tiles.
        for(int rb=0;rb<32;rb+=4)for(int k=0;k<256;k+=16)for(int r=0;r<4;++r) {
            Axis v;v.keep=-1;v.strb=-1;v.last=(rb==28&&k==240&&r==3);
            for(int t=0;t<16;++t)v.data.range(8*t+7,8*t)=(uint8_t)a[(rb+r)*256+k+t];ia.write(v);
        }
        for(int cb=0;cb<32;cb+=8)for(int k=0;k<256;k+=2) {
            Axis v;v.keep=-1;v.strb=-1;v.last=(cb==24&&k==254);
            for(int t=0;t<16;++t)v.data.range(8*t+7,8*t)=(uint8_t)b[(k+t/8)*32+cb+t%8];ib.write(v);
        }
        for(int rb=0;rb<32;rb+=4)for(int cb=0;cb<32;cb+=8)for(int r=0;r<4;++r)for(int j=0;j<8;j+=4){
            Axis v=ic.read();for(int t=0;t<4;++t)c[(rb+r)*32+cb+j+t]=(int32_t)(uint32_t)v.data.range(32*t+31,32*t);
        }
    }
    void bfloat(const float* a,const float* b,float* c) {
        for(int rb=0;rb<32;rb+=4)for(int k=0;k<64;k+=8)for(int r=0;r<4;++r) {
            Axis v;v.keep=-1;v.strb=-1;v.last=rb==28&&k==56&&r==3;
            for(int t=0;t<8;++t){union{float f;uint32_t u;}x;x.f=bert_aie::bf16(a[(rb+r)*64+k+t]);v.data.range(16*t+15,16*t)=x.u>>16;}fa.write(v);
        }
        for(int cb=0;cb<32;cb+=8)for(int k=0;k<64;++k) {
            Axis v;v.keep=-1;v.strb=-1;v.last=cb==24&&k==63;
            for(int t=0;t<8;++t){union{float f;uint32_t u;}x;x.f=bert_aie::bf16(b[k*32+cb+t]);v.data.range(16*t+15,16*t)=x.u>>16;}fb.write(v);
        }
        for(int rb=0;rb<32;rb+=4)for(int cb=0;cb<32;cb+=8)for(int r=0;r<4;++r)for(int j=0;j<8;j+=4){Axis v=fc.read();for(int t=0;t<4;++t){union{float f;uint32_t u;}x;x.u=(uint32_t)v.data.range(32*t+31,32*t);c[(rb+r)*32+cb+j+t]=x.f;}}
    }
};
#ifndef BERT_RESIDENT
#define BERT_RESIDENT 1
#endif
#ifndef BERT_WIRE_TEST_ONLY
extern "C" void bert_pl(const int32_t* inputs,const float* parameters,const int8_t* weights,
    const float* scales,const float* input_scales,float* scratch,float* output,
    Stream& ia,Stream& ib,Stream& ic,Stream& fa,Stream& fb,Stream& fc) {
#pragma HLS INTERFACE m_axi port=inputs offset=slave bundle=inputs depth=768
#pragma HLS INTERFACE m_axi port=parameters offset=slave bundle=parameters depth=8025346
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=weights depth=3145728 max_read_burst_length=256 num_read_outstanding=16 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=scales offset=slave bundle=parameters depth=9216
#pragma HLS INTERFACE m_axi port=input_scales offset=slave bundle=parameters depth=24
#pragma HLS INTERFACE m_axi port=scratch offset=slave bundle=scratch depth=720896
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=inputs depth=3
#pragma HLS INTERFACE s_axilite port=inputs bundle=control
#pragma HLS INTERFACE s_axilite port=parameters bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=scales bundle=control
#pragma HLS INTERFACE s_axilite port=input_scales bundle=control
#pragma HLS INTERFACE s_axilite port=scratch bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE axis port=ia
#pragma HLS INTERFACE axis port=ib
#pragma HLS INTERFACE axis port=ic
#pragma HLS INTERFACE axis port=fa
#pragma HLS INTERFACE axis port=fb
#pragma HLS INTERFACE axis port=fc
    AieTiles e{ia,ib,ic,fa,fb,fc};
    int status=bert_aie::model<BERT_RESIDENT>(e,inputs,inputs+256,inputs+512,
        parameters,weights,scales,input_scales,scratch,output);
    output[2]=float(status);
}
#endif
