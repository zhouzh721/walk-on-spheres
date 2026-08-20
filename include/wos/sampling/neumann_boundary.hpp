#pragma once

#include "wos/boundary/scene.hpp"
#include "wos/prng.hpp"

namespace wos {

struct NeumannBoundarySample2D {
    bool valid = false;
    Point2D point{};
    Point2D normal{};
    int primitive_id = -1;
    int boundary_id = -1;
    double pdf = 0.0;
};

// Uniformly sample arclength from the portions of Neumann segments contained
// in the disk. The PDF is with respect to boundary arclength.
NeumannBoundarySample2D sample_neumann_boundary(
    const BoundaryScene2D &scene, Point2D center, double radius, PRNG &rng);

} // namespace wos
