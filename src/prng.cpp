#include "wos/hash.hpp"
#include "wos/prng.hpp"

namespace wos {

PRNG::PRNG(std::uint64_t seed) {
    // expand one 64-bit seed into the 4-word xoshiro state via splitmix64
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(seed + 0x9E3779B97F4A7C15ULL);
    state_[2] = splitmix64(seed + 0x3C6EF372FE94F82AULL);
    state_[3] = splitmix64(seed + 0xDAA66D2C7DDF743FULL);

    // xoshiro requires non-zero state
    if (!(state_[0] | state_[1] | state_[2] | state_[3])) state_[0] = 1;
}

}
