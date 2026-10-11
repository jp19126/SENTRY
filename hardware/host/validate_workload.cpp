// Offline packet decoder for the selected Windows-to-A72 workload protocol.
// No MMIO, device opens or simulated detector arithmetic.
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static uint32_t u32(const uint8_t *p){return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;}
static uint64_t u64(const uint8_t *p){return uint64_t(u32(p))|uint64_t(u32(p+4))<<32;}
int main(int argc,char **argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: validate_workload workload.bin");
  std::ifstream in(argv[1],std::ios::binary);if(!in)throw std::runtime_error("cannot open workload");
  std::array<uint8_t,64> h{};uint64_t packets=0,documents=0,active_doc=0,model=0;uint32_t expected=0,threshold=0,fixed_length=0;bool active=false;
  while(in.read(reinterpret_cast<char*>(h.data()),h.size())){
   const auto *p=h.data();uint32_t length=u32(p+12),valid=u32(p+16),index=u32(p+20),flags=u32(p+48),bytes=u32(p+52);
   uint64_t doc=u64(p+24),sample=u64(p+32),current_model=u64(p+40);(void)sample;
   if(std::memcmp(p,"SENTRY01",8)||u32(p+8)!=1||u32(p+60)!=0)throw std::runtime_error("header/version/reserved");
   if((length!=128&&length!=256&&length!=512)||valid<1||valid>length||bytes!=9*length||flags>1)throw std::runtime_error("shape/payload/flags");
   if(packets && (current_model!=model||u32(p+56)!=threshold||length!=fixed_length))throw std::runtime_error("cold configuration changed");
   fixed_length=length;model=current_model;threshold=u32(p+56);if(threshold>0x3f800000)throw std::runtime_error("threshold");
   if(!active){if(index!=0)throw std::runtime_error("new document index");active=true;active_doc=doc;expected=0;++documents;}
   if(doc!=active_doc||index!=expected++)throw std::runtime_error("document/window association");
   std::vector<uint8_t> payload(bytes);if(!in.read(reinterpret_cast<char*>(payload.data()),bytes))throw std::runtime_error("truncated payload");
   uint32_t count=0;for(uint32_t i=0;i<length;++i){
    if(u32(payload.data()+4*i)>=30522||u32(payload.data()+4*length+4*i)>1||payload[8*length+i]!=(i<valid?1:0))throw std::runtime_error("ID/type/mask range");
    count+=payload[8*length+i];
   }
   if(count!=valid)throw std::runtime_error("valid length/mask mismatch");
   if(flags&1){active=false;}
   ++packets;
  }
  if(in.gcount()!=0||active||!packets)throw std::runtime_error("truncated header/unclosed document/empty input");
  std::cout<<"{\"passed\":true,\"packets\":"<<packets<<",\"documents\":"<<documents<<",\"detector_execution\":false}\n";
  return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
