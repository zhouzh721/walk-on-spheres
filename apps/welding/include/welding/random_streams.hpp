#pragma once

#include <cstddef>
#include <cstdint>

#include "wos/hash.hpp"
#include "wos/prng.hpp"

namespace welding {

struct PathRandomStreams {
    wos::PRNG walk;
    wos::PRNG source;
};

inline std::uint64_t point_seed(std::uint64_t base_seed,
                                std::uint64_t stream_key,
                                std::size_t point_index) {
    return wos::splitmix64(
        base_seed ^ stream_key ^
        wos::splitmix64(static_cast<std::uint64_t>(point_index)));
}

inline PathRandomStreams make_path_random_streams(
        std::uint64_t point_seed_value) {
    return PathRandomStreams{
        wos::PRNG(wos::splitmix64(
            point_seed_value ^ 0x13198A2E03707344ULL)),
        wos::PRNG(wos::splitmix64(
            point_seed_value ^ 0xA4093822299F31D0ULL)),
    };
}

} // namespace welding
