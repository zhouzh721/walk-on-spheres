// xoshiro256++ pseudo-random number generator with explicit per-instance state
#pragma once
#include <cstdint>

namespace wos {

class PRNG {
public:
    explicit PRNG(std::uint64_t seed);

    // Raw 64-bit uniform random integer.
    std::uint64_t u64() {
        const std::uint64_t result = rotl64(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl64(state_[3], 45);
        return result;
    }

    // Uniform double in [0, 1), using the top 53 bits.
    double unit() {
        return (u64() >> 11) * (1.0 / (1ULL << 53));
    }

    // Uniform double in (0, 1). Use this when an exact zero would map to a
    // singular point, for example the centre of a source-sampling sphere.
    double unit_open() {
        double value;
        do {
            value = unit();
        } while (value == 0.0);
        return value;
    }

private:
    static std::uint64_t rotl64(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    std::uint64_t state_[4];
};

}
