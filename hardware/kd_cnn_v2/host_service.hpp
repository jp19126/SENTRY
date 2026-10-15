#ifndef KD_HOST_SERVICE_HPP
#define KD_HOST_SERVICE_HPP
#include "host_driver.hpp"
namespace kdhost {
// Single trusted sender; IDs/masks must be the established tokenizer's complete
// document windows. SHA256 binds transport bytes, not tokenizer semantics.
struct Wire {void *context;bool (*receive)(void*,uint8_t*,size_t);bool (*send)(void*,const uint8_t*,size_t);};
struct Storage {uint8_t *model,*raw;uint16_t *ids;uint8_t *masks;uint32_t max_raw,max_windows;};
class GuardService {
 Wire wire;Storage data;Driver& driver;int32_t threshold;uint32_t polls,next=1;bool ready=false,failed=false;uint8_t model_sha[32]={};
 bool reply(uint32_t,uint32_t,int,const Decision*);
 public:GuardService(Wire w,Storage s,Driver& d,int32_t t,uint32_t p):wire(w),data(s),driver(d),threshold(t),polls(p){}
 bool receive_model(const uint8_t expected_sha[32]);
 // False is terminal: preserve retained data and require operator recovery.
 bool document();
};
}
#endif
