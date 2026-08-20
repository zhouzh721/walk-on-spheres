#pragma once

#include "wos/mesh.hpp"

namespace wos {

struct NeumannSilhouetteResult2D {
    bool found = false;
    double distance = 0.0;
    Point2D point{};
    int primitive_id = -1;
    int boundary_id = -1;
};

} // namespace wos
