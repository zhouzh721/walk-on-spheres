#pragma once

#include "wos/mesh.hpp"

namespace wos {

template<int N>
struct NearestPointResult {
    double distance;
    Point<N> point;
    int primitive_id;
    int boundary_id;
};

} // namespace wos
