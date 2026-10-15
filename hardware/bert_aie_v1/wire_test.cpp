// Actual PL adapter's packing test using vendor AXIS types; no device access.
#define BERT_WIRE_TEST_ONLY
#include "pl.cpp"
#include <cassert>
#include <iostream>
int main() {
    Stream ia,ib,ic,fa,fb,fc;AieTiles wire{ia,ib,ic,fa,fb,fc};bert_aie::CpuTiles cpu;
    int8_t a[8192],b[8192];int32_t expected[1024],actual[1024];
    for(int n=0;n<8192;++n){a[n]=int8_t((n*13)%255-127);b[n]=int8_t((n*19)%255-127);}
    cpu.i8(a,b,expected);
    for(int rb=0;rb<32;rb+=4)for(int cb=0;cb<32;cb+=8)for(int r=0;r<4;++r)for(int j=0;j<8;j+=4) {
        Axis v;for(int t=0;t<4;++t)v.data.range(32*t+31,32*t)=uint32_t(expected[(rb+r)*32+cb+j+t]);ic.write(v);
    }
    wire.i8(a,b,actual);
    for(int n=0;n<1024;++n)assert(actual[n]==expected[n]);
    for(int rb=0;rb<32;rb+=4)for(int k=0;k<256;k+=16)for(int r=0;r<4;++r) {
        Axis v=ia.read();assert(bool(v.last)==(rb==28&&k==240&&r==3));
        for(int t=0;t<16;++t)assert(uint8_t(v.data.range(8*t+7,8*t))==uint8_t(a[(rb+r)*256+k+t]));
    }
    for(int cb=0;cb<32;cb+=8)for(int k=0;k<256;k+=2) {
        Axis v=ib.read();assert(bool(v.last)==(cb==24&&k==254));
        for(int t=0;t<16;++t)assert(uint8_t(v.data.range(8*t+7,8*t))==uint8_t(b[(k+t/8)*32+cb+t%8]));
    }
    float aa[2048],bb[2048],cc[1024],dd[1024];
    for(int n=0;n<2048;++n)aa[n]=float(n-1024)/73.f;
    for(int n=0;n<2048;++n)bb[n]=float(n-1024)/91.f;
    cpu.bfloat(aa,bb,cc);
    for(int rb=0;rb<32;rb+=4)for(int cb=0;cb<32;cb+=8)for(int r=0;r<4;++r)for(int j=0;j<8;j+=4){Axis v;for(int t=0;t<4;++t){union{float f;uint32_t u;}x;x.f=cc[(rb+r)*32+cb+j+t];v.data.range(32*t+31,32*t)=x.u;}fc.write(v);}
    wire.bfloat(aa,bb,dd);
    for(int n=0;n<1024;++n)assert(dd[n]==cc[n]);
    for(int rb=0;rb<32;rb+=4)for(int k=0;k<64;k+=8)for(int r=0;r<4;++r){Axis v=fa.read();for(int t=0;t<8;++t){union{float f;uint32_t u;}x;x.f=bert_aie::bf16(aa[(rb+r)*64+k+t]);assert(uint16_t(v.data.range(16*t+15,16*t))==uint16_t(x.u>>16));}}
    for(int cb=0;cb<32;cb+=8)for(int k=0;k<64;++k){Axis v=fb.read();for(int t=0;t<8;++t){union{float f;uint32_t u;}x;x.f=bert_aie::bf16(bb[k*32+cb+t]);assert(uint16_t(v.data.range(16*t+15,16*t))==uint16_t(x.u>>16));}}
    assert(ia.empty()&&ib.empty()&&ic.empty()&&fa.empty()&&fb.empty()&&fc.empty());
    std::cout<<"PASS actual PL AXIS packing/roundtrip: int8 32x256x32, BF16 32x64x32. Arithmetic response injected from independent portable service; not AIE simulation.\n";
}
