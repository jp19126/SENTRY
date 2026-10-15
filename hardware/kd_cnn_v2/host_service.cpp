#include "host_service.hpp"
#include "host_sha256.hpp"
#include <cstring>
namespace kdhost {
static uint32_t le(const uint8_t*p){return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;}
static void put(uint8_t*p,uint32_t x){for(unsigned i=0;i<4;++i)p[i]=uint8_t(x>>(8*i));}
bool GuardService::reply(uint32_t seq,uint32_t status,int error,const Decision*d){uint8_t out[64]={};std::memcpy(out,"KDR2",4);put(out+4,seq);put(out+8,status);put(out+12,uint32_t(error));if(d){put(out+16,uint32_t(d->margin));put(out+20,d->cost.windows);put(out+24,d->cost.chunks);put(out+28,d->cost.cold);const uint64_t sizes[3]={d->cost.model_write_bytes,d->cost.input_write_bytes,d->cost.output_read_bytes};for(unsigned i=0;i<3;++i){put(out+32+8*i,uint32_t(sizes[i]));put(out+36+8*i,uint32_t(sizes[i]>>32));}}return wire.send(wire.context,out,sizeof out);}
bool GuardService::receive_model(const uint8_t expected[32]){
 if(ready||failed)return false;
 uint8_t actual[32];if(!wire.receive(wire.context,data.model,kdcnn::PARAM_BYTES)){failed=true;reply(0,2,20,nullptr);return false;}
 Sha256 h;h.update(data.model,kdcnn::PARAM_BYTES);h.finish(actual);
 if(std::memcmp(actual,expected,32)){failed=true;reply(0,2,21,nullptr);return false;}
 std::memcpy(model_sha,expected,32);uint8_t greeting[64]={};std::memcpy(greeting,"KDR2",4);put(greeting+8,3);std::memcpy(greeting+16,expected,32);put(greeting+48,uint32_t(threshold));ready=wire.send(wire.context,greeting,sizeof greeting);failed=!ready;return ready;
}
bool GuardService::document(){
 if(!ready||failed)return false;
 uint8_t header[84];uint32_t seq=next;int error=22;
 if(!wire.receive(wire.context,header,sizeof header))goto fail;
 {uint32_t rawbytes=le(header+8),windows=le(header+12);seq=le(header+4);
 if(std::memcmp(header,"KDC2",4)||std::memcmp(header+16,model_sha,32)||le(header+48)!=uint32_t(threshold)||seq!=next||!next||rawbytes>data.max_raw||!windows||windows>data.max_windows){error=23;goto fail;}
 size_t idbytes=size_t(windows)*kdcnn::L*2,maskbytes=size_t(windows)*kdcnn::L;
 if(!wire.receive(wire.context,data.raw,rawbytes)||!wire.receive(wire.context,(uint8_t*)data.ids,idbytes)||!wire.receive(wire.context,data.masks,maskbytes))goto fail;
 Sha256 h;uint8_t actual[32];h.update(header,52);h.update(data.raw,rawbytes);h.update(data.ids,idbytes);h.update(data.masks,maskbytes);h.finish(actual);
 if(std::memcmp(actual,header+52,32)){error=24;goto fail;}
 auto result=driver.run(data.model,kdcnn::PARAM_BYTES,data.ids,data.masks,windows,threshold,data.raw,rawbytes,polls);
 if(!result.valid){failed=true;reply(seq,2,result.error,&result);return false;}
 if(!reply(seq,result.blocked?1:0,0,&result)){failed=true;return false;}
 // The only raw-document transmission is after final valid allow.
 if(!result.blocked&&!wire.send(wire.context,data.raw,rawbytes)){failed=true;return false;}
 ++next;return true;
 }
 fail:failed=true;reply(seq,2,error,nullptr);return false;
}
}
