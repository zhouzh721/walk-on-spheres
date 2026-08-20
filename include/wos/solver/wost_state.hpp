#pragma once

#include "wos/mesh.hpp"

namespace wos::solver {

struct WoStState2D {
    Point2D position{};
    double value = 0.0;
    double throughput = 1.0;
    int steps = 0;
    bool on_neumann = false;
    Point2D neumann_normal{};
    Point2D previous_direction{};
};

} // namespace wos::solver
