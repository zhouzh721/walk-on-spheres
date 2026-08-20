#pragma once

#include <stdexcept>

#include "wos/boundary/type.hpp"

namespace wos {

// Boundary data use the following convention:
//   Dirichlet: u = value
//   Neumann:   n dot (a grad u) = value
//   Robin:     n dot (a grad u) + coefficient * u = value
struct BoundaryCondition {
    BoundaryType type;
    double value;
    double coefficient;

    static constexpr BoundaryCondition dirichlet(double value) {
        return BoundaryCondition{BoundaryType::Dirichlet, value, 0.0};
    }

    static constexpr BoundaryCondition neumann(double flux) {
        return BoundaryCondition{BoundaryType::Neumann, flux, 0.0};
    }

    static constexpr BoundaryCondition robin(double coefficient,
                                              double value) {
        return BoundaryCondition{BoundaryType::Robin, value, coefficient};
    }
};

inline double dirichlet_value(const BoundaryCondition &condition) {
    if (condition.type != BoundaryType::Dirichlet) {
        throw std::invalid_argument(
            "WoS supports Dirichlet boundaries only; use WoSt for mixed boundary conditions");
    }
    return condition.value;
}

} // namespace wos
