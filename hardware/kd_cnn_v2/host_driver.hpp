#ifndef SENTRY_KD_CNN_HOST_HPP
#define SENTRY_KD_CNN_HOST_HPP
#include "cnn.hpp"
#include <cstddef>
namespace kdhost {
// Offsets must come from the actual exported HLS driver/component, never from
// the synthetic test's map. Each pointer offset names a contiguous 64-bit pair.
struct Registers {uint32_t control,blob,bytes,ids,masks,count,load,first,final,
 previous,previous_windows,threshold,output;};
struct Buffers {uint64_t model,ids,masks,output;};
struct Platform {
 void *context;
 void (*write32)(void*,uint64_t,uint32_t);
 uint32_t (*read32)(void*,uint64_t);
 bool (*write_memory)(void*,uint64_t,const void*,size_t);
 bool (*read_memory)(void*,uint64_t,void*,size_t);
 void (*flush)(void*,uint64_t,size_t);
 void (*invalidate)(void*,uint64_t,size_t);
 void (*barrier)(void*);
 // Called only after every window and final result passes. The caller must
 // retain the raw document until this callback (or retain it on every error).
 void (*release_document)(void*,const uint8_t*,size_t,bool);
};
struct Accounting {uint64_t model_write_bytes=0,input_write_bytes=0,output_read_bytes=0;
 uint32_t chunks=0,windows=0,polls=0;bool cold=false;};
struct Decision {int32_t margin=0;bool blocked=false,valid=false;int error=0;Accounting cost;};
class Driver {
 Platform io;Registers regs;Buffers buffers;uint64_t control;
 bool loaded=false,faulted=false;
 void set(uint32_t,uint32_t);void pointer(uint32_t,uint64_t);
 public:
 Driver(Platform p,Registers r,Buffers b,uint64_t c):io(p),regs(r),buffers(b),control(c){}
 // Call only after an external, explicitly authorized device reset completes.
 void after_device_reset(){loaded=false;faulted=false;}
 Decision run(const uint8_t *verified_model,uint32_t model_bytes,const uint16_t *ids,
  const uint8_t *masks,uint32_t windows,int32_t threshold,const uint8_t *raw,
  size_t raw_bytes,uint32_t poll_limit);
};
}
#endif
