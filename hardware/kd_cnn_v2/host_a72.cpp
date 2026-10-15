#include "host_a72.hpp"
#include "xil_io.h"
#include "xil_cache.h"
#include <cstring>
namespace kdhost {
static bool span(A72Context& c,uint64_t a,size_t n){return a>=c.ddr_base&&a-c.ddr_base<=c.ddr_bytes&&n<=c.ddr_bytes-(a-c.ddr_base);}
static void write32(void*,uint64_t a,uint32_t v){Xil_Out32(UINTPTR(a),v);}
static uint32_t read32(void*,uint64_t a){return Xil_In32(UINTPTR(a));}
static bool write_memory(void *p,uint64_t a,const void *v,size_t n){if(!span(*(A72Context*)p,a,n))return false;std::memcpy(reinterpret_cast<void*>(UINTPTR(a)),v,n);return true;}
static bool read_memory(void *p,uint64_t a,void *v,size_t n){if(!span(*(A72Context*)p,a,n))return false;std::memcpy(v,reinterpret_cast<const void*>(UINTPTR(a)),n);return true;}
static void flush(void*,uint64_t a,size_t n){Xil_DCacheFlushRange(UINTPTR(a),n);}
static void invalidate(void*,uint64_t a,size_t n){Xil_DCacheInvalidateRange(UINTPTR(a),n);}
static void barrier(void*){asm volatile("dsb sy" ::: "memory");}
static void release(void *p,const uint8_t *raw,size_t n,bool blocked){A72Context& c=*(A72Context*)p;c.release(c.user,raw,n,blocked);}
Platform a72_platform(A72Context& c){return Platform{&c,write32,read32,write_memory,read_memory,flush,invalidate,barrier,release};}
}
