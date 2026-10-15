#include "host_driver.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
using namespace kdhost;
// Deliberately synthetic offsets. No board address or actual IP-map claim.
#ifdef CNN_REGISTER_MAP_HEADER
#include CNN_REGISTER_MAP_HEADER
const Registers regs=kdhost::synthesized_registers;
#else
const Registers regs={0,16,32,48,64,80,96,112,128,144,160,176,192};
#endif
const Buffers buffers={0x40000000,0x40300000,0x40302000,0x40304000};
constexpr uint64_t CONTROL=0xA4000000,DDR=0x40000000;
struct Fake {std::vector<uint8_t> memory=std::vector<uint8_t>(4*1024*1024);uint32_t reg[64]={};bool timeout=false,corrupt=false;int releases=0,calls=0,flushes=0,invalidates=0,barriers=0;bool saw_partial_release=false;};
static uint64_t ptr(Fake& f,uint32_t r){return uint64_t(f.reg[r/4])|(uint64_t(f.reg[r/4+1])<<32);}
static void wr(void *p,uint64_t address,uint32_t value){
 Fake& f=*(Fake*)p;auto r=address-CONTROL;if(r>=256)throw std::runtime_error("register bounds");f.reg[r/4]=value;
 if(r==regs.control&&(value&1)){
  if(f.timeout){f.reg[0]=0;return;}++f.calls;
  auto memory=[&](uint32_t r){auto address=ptr(f,r);if(address<DDR||address>=DDR+f.memory.size())throw std::runtime_error("DDR bounds");return f.memory.data()+address-DDR;};
  auto *out=reinterpret_cast<kdcnn::Result*>(memory(regs.output));
  kd_cnn_top(memory(regs.blob),f.reg[regs.bytes/4],reinterpret_cast<uint16_t*>(memory(regs.ids)),memory(regs.masks),f.reg[regs.count/4],f.reg[regs.load/4],f.reg[regs.first/4],f.reg[regs.final/4],int32_t(f.reg[regs.previous/4]),f.reg[regs.previous_windows/4],int32_t(f.reg[regs.threshold/4]),out);
  if(f.corrupt)out->windows+=1;f.reg[0]=6;
 }
}
static uint32_t rd(void *p,uint64_t a){Fake& f=*(Fake*)p;return f.reg[(a-CONTROL)/4];}
static bool memwrite(void *p,uint64_t a,const void *v,size_t n){Fake& f=*(Fake*)p;if(a<DDR||a-DDR+n>f.memory.size())return false;std::memcpy(f.memory.data()+a-DDR,v,n);return true;}
static bool memread(void *p,uint64_t a,void *v,size_t n){Fake& f=*(Fake*)p;if(a<DDR||a-DDR+n>f.memory.size())return false;std::memcpy(v,f.memory.data()+a-DDR,n);return true;}
static void flush(void*p,uint64_t,size_t){++((Fake*)p)->flushes;}
static void invalidate(void*p,uint64_t,size_t){++((Fake*)p)->invalidates;}
static void barrier(void*p){++((Fake*)p)->barriers;}
static void release(void*p,const uint8_t *raw,size_t n,bool){Fake& f=*(Fake*)p;if(n!=8||std::memcmp(raw,"retained",8))throw std::runtime_error("document not retained");if(f.reg[regs.final/4]!=1)f.saw_partial_release=true;++f.releases;}
template<class T>T read(std::ifstream&f){T x;f.read((char*)&x,sizeof(x));return x;}
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream model(argv[1],std::ios::binary);std::vector<uint8_t> blob((std::istreambuf_iterator<char>(model)),{});
 std::ifstream cases(argv[2],std::ios::binary);read<uint32_t>(cases);uint32_t count=read<uint32_t>(cases);read<uint32_t>(cases);read<uint32_t>(cases);read<int32_t>(cases);read<uint32_t>(cases);read<int32_t>(cases);int32_t expected=read<int32_t>(cases);read<int32_t>(cases);if(count!=1)return 3;
 std::vector<uint16_t> first(kdcnn::L);std::vector<uint8_t> mask(kdcnn::L);cases.read((char*)first.data(),kdcnn::L*2);cases.read((char*)mask.data(),kdcnn::L);if(!cases)return 4;
 std::vector<uint16_t> ids;std::vector<uint8_t> masks;for(int i=0;i<17;++i){ids.insert(ids.end(),first.begin(),first.end());masks.insert(masks.end(),mask.begin(),mask.end());}
 Fake f;Platform io={&f,wr,rd,memwrite,memread,flush,invalidate,barrier,release};Driver driver(io,regs,buffers,CONTROL);const uint8_t raw[]="retained";
 auto cold=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),17,expected,raw,8,8);
 if(!cold.valid||cold.blocked||cold.margin!=expected||cold.cost.chunks!=2||cold.cost.windows!=17||cold.cost.model_write_bytes!=blob.size()||cold.cost.input_write_bytes!=17*256*3||cold.cost.output_read_bytes!=48||f.releases!=1||f.saw_partial_release)return 5;
 auto warm=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected-1,raw,8,8);
 if(!warm.valid||!warm.blocked||warm.cost.cold||warm.cost.model_write_bytes||f.releases!=2)return 6;
 f.corrupt=true;auto invalid=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected,raw,8,8);if(invalid.valid||invalid.error!=15||f.releases!=2)return 7;
 auto fault=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected,raw,8,8);if(fault.error!=10)return 8;
 driver.after_device_reset();f.corrupt=false;f.timeout=true;auto timeout=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected,raw,8,8);if(timeout.error!=14||timeout.cost.polls!=8||f.releases!=2)return 9;
 driver.after_device_reset();f.timeout=false;auto resumed=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected,raw,8,8);if(!resumed.valid||!resumed.cost.cold||f.releases!=3)return 10;
 masks[0]=2;auto badinput=driver.run(blob.data(),blob.size(),ids.data(),masks.data(),1,expected,raw,8,8);if(badinput.error!=12||f.releases!=3)return 11;
 std::cout<<"{\"passed\":true,\"cases\":7,\"cold_model_bytes\":"<<cold.cost.model_write_bytes<<",\"cold_input_bytes\":"<<cold.cost.input_write_bytes<<",\"cold_output_bytes\":"<<cold.cost.output_read_bytes<<",\"warm_model_bytes\":"<<warm.cost.model_write_bytes<<",\"chunks\":"<<cold.cost.chunks<<",\"windows\":"<<cold.cost.windows<<",\"releases\":"<<f.releases<<",\"flush_calls\":"<<f.flushes<<",\"invalidate_calls\":"<<f.invalidates<<",\"barrier_calls\":"<<f.barriers<<",\"evidence\":\"CPU fake-register/memory lifecycle invoking real complete C CNN; not RTL, DDR or board timing\"}\n";
 return 0;
}
