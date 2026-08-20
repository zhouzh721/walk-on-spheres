#pragma once

#include "wos/mesh.hpp"

namespace wos {

struct RayHit2D {
    bool hit = false;
    double distance = 0.0;
    Point2D point{};
    Point2D normal{};
    int primitive_id = -1;
    int boundary_id = -1;
};

} // namespace wos
