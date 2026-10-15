#ifndef KD_HOST_SHA256_HPP
#define KD_HOST_SHA256_HPP
#include <cstdint>
#include <cstddef>
namespace kdhost {
class Sha256 {
 uint32_t h[8];uint8_t block[64];uint64_t bytes=0;unsigned used=0;
 void compress();
 public:Sha256();void update(const void*,size_t);void finish(uint8_t[32]);
};
}
#endif
