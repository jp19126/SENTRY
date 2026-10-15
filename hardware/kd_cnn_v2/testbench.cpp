#include "cnn.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <climits>
#include <cmath>
template<class T> T read(std::ifstream& f){T x;f.read(reinterpret_cast<char*>(&x),sizeof(x));if(!f)throw 1;return x;}
int main(int argc,char **argv){
 if(argc!=3){std::cerr<<"model.bin cases.bin\n";return 2;}
 int rounding=0;
 for(int s: {-62,-31,-12,-3,-1,0,1,7,31})for(int64_t x: {-(int64_t(1)<<55),-(int64_t(1)<<55)+1,int64_t(-99999),int64_t(-129),int64_t(-5),int64_t(-3),int64_t(-1),int64_t(0),int64_t(1),int64_t(3),int64_t(5),int64_t(129),int64_t(99999),(int64_t(1)<<55)-1,int64_t(1)<<55}){
  long double want=std::nearbyintl(std::ldexp((long double)x,s));want=want>127?127:want< -127?-127:want;
  if(kdcnn::rne_shift(x,s,127)!=int64_t(want))return 6;++rounding;
 }
 for(int d:{1,2,3,4,255,256})for(int x=-511;x<=511;++x){if(kdcnn::rne_div(x,d)!=int64_t(std::nearbyintl((long double)x/d)))return 7;++rounding;}
 std::ifstream f(argv[1],std::ios::binary);std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)),{});
 if(blob.size()!=kdcnn::PARAM_BYTES)return 3;
 // HLS co-simulation serializes each pointer's declared AXI depth, including
 // padding beyond the active byte/window count. Allocate that complete extent.
 blob.resize(2040000,0);
 std::ifstream cases(argv[2],std::ios::binary);const auto n=read<uint32_t>(cases);
 for(uint32_t i=0;i<n;++i){
  const auto count=read<uint32_t>(cases),first=read<uint32_t>(cases),final=read<uint32_t>(cases);
  const auto prev=read<int32_t>(cases);const auto prevwindows=read<uint32_t>(cases);const auto threshold=read<int32_t>(cases);
  const auto expected=read<int32_t>(cases);const auto blocked=read<int32_t>(cases);
  std::vector<uint16_t> ids(kdcnn::MAX_WINDOWS*kdcnn::L);std::vector<uint8_t> mask(kdcnn::MAX_WINDOWS*kdcnn::L);
  cases.read(reinterpret_cast<char*>(ids.data()),count*kdcnn::L*2);cases.read(reinterpret_cast<char*>(mask.data()),count*kdcnn::L);
  if(!cases)return 4;kdcnn::Result out;
  kd_cnn_top(blob.data(),kdcnn::PARAM_BYTES,ids.data(),mask.data(),count,i==0,first,final,prev,prevwindows,threshold,&out);
  std::cout<<i<<" "<<out.margin<<" "<<out.maximum<<" "<<out.blocked<<" "<<out.error<<" "<<out.windows<<" "<<out.decision_valid<<"\n";
  if(out.error||out.maximum!=expected||out.blocked!=blocked||out.windows!=(first?count:prevwindows+count)||out.decision_valid!=final)return 5;
 }
 // Failure/reset behavior at the actual public interface; no invalid array read.
 uint16_t ids[kdcnn::MAX_WINDOWS*kdcnn::L]={};uint8_t mask[kdcnn::MAX_WINDOWS*kdcnn::L]={};kdcnn::Result out;
 auto invoke=[&](uint32_t bytes,uint32_t count,uint32_t load,uint32_t first,uint32_t previous_windows){kd_cnn_top(blob.data(),bytes,ids,mask,count,load,first,1,0,previous_windows,0,&out);};
 invoke(kdcnn::PARAM_BYTES,17,0,1,0);if(out.error!=3||out.decision_valid)return 8;
 mask[0]=2;invoke(kdcnn::PARAM_BYTES,1,0,1,0);if(out.error!=2||out.decision_valid)return 9;
 mask[0]=1;ids[0]=65535;invoke(kdcnn::PARAM_BYTES,1,0,1,0);if(out.error!=2||out.decision_valid)return 10;
 mask[0]=0;invoke(kdcnn::PARAM_BYTES,1,0,1,0);if(out.error||!out.decision_valid)return 11;
 invoke(kdcnn::PARAM_BYTES,1,0,0,UINT_MAX);if(out.error!=3)return 12;
 invoke(kdcnn::PARAM_BYTES-1,1,1,1,0);if(out.error!=1)return 13;
 invoke(kdcnn::PARAM_BYTES,1,0,1,0);if(out.error!=1)return 14;
 invoke(kdcnn::PARAM_BYTES,1,1,1,0);if(out.error||!out.decision_valid||out.windows!=1)return 15;
 invoke(kdcnn::PARAM_BYTES,1,1,0,1);if(out.error!=3)return 16;
 std::cout<<"ROUNDING "<<rounding<<" INTERFACE 9\n";return 0;
}
