#pragma once

#include <stdexcept>

#include "wos/prng.hpp"
#include "wos/solver/result.hpp"
#include "wos/solver/settings.hpp"
#include "wos/solver/start_point.hpp"

namespace wos::solver {

// Public API placeholder for Walk on Stars. The geometric star-domain walk
// and mixed-boundary estimators are intentionally implemented in a later
// stage. WoSt is not exposed by the command-line application yet.
class WoSt {
public:
    explicit WoSt(Settings settings);

    const Settings &settings() const {
        return settings_;
    }

    template<int N, typename Equation>
    Result solve(const BVH<N> &, Point<N>, const StartPoint<N> &,
                 const Equation &, PRNG &, PRNG &) const {
        throw std::logic_error("WoSt is declared but not implemented");
    }

    template<int N, typename Equation>
    Result solve(const BVH<N> &bvh, Point<N> point,
                 const Equation &equation, PRNG &walk_rng,
                 PRNG &source_rng) const {
        return solve(bvh, point, find_start_point(bvh, point), equation,
                     walk_rng, source_rng);
    }

private:
    Settings settings_;
};

} // namespace wos::solver
