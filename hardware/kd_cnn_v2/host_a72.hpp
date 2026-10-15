#ifndef SENTRY_KD_CNN_A72_HPP
#define SENTRY_KD_CNN_A72_HPP
#include "host_driver.hpp"
namespace kdhost {
struct A72Context {uint64_t ddr_base,ddr_bytes;void *user;void (*release)(void*,const uint8_t*,size_t,bool);};
// Constructing this adapter performs no MMIO. Running Driver through it is a
// physical operation and requires separately approved deployment/access.
Platform a72_platform(A72Context&);
}
#endif
