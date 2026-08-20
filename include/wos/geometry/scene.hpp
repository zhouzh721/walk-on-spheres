#pragma once

#include "wos/geometry/classification.hpp"
#include "wos/geometry/nearest_point.hpp"
#include "wos/prng.hpp"

namespace wos {

// Dimension-independent geometry API used by solvers and applications.
// Concrete acceleration-library types must not leak through this interface.
template<int N>
class GeometryScene {
public:
    virtual ~GeometryScene() = default;

    virtual const Mesh<N> &mesh() const = 0;
    virtual NearestPointResult<N> closest_boundary(
        Point<N> point) const = 0;

    virtual PointClassification classify_point(
        Point<N> point, double boundary_distance,
        double boundary_tolerance, int max_ray_attempts,
        PRNG &rng) const = 0;
};

} // namespace wos
