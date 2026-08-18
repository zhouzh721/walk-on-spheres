#pragma once

#include <cstddef>
#include <stdexcept>

#include "wos/grid.hpp"
#include "wos/mesh.hpp"

namespace welding {

inline wos::Mesh<2> make_rectangular_plate(double xmin, double xmax,
                                            double ymin, double ymax) {
    if (!(xmax > xmin) || !(ymax > ymin)) {
        throw std::invalid_argument(
            "rectangular plate requires increasing coordinate bounds");
    }

    wos::Mesh<2> mesh;
    mesh.verts = {
        {xmin, ymin},
        {xmax, ymin},
        {xmax, ymax},
        {xmin, ymax},
    };
    mesh.prims = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
    };
    mesh.boundary_ids = {0, 0, 0, 0};
    return mesh;
}

inline std::size_t flat_index_2d(int i, int j, int ny) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(ny)
         + static_cast<std::size_t>(j);
}

inline bool is_boundary_index_2d(const wos::Grid &grid, int i, int j) {
    return i == 0 || i == grid.Nx - 1 || j == 0 || j == grid.Ny - 1;
}

} // namespace welding
