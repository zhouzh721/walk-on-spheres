#pragma once

#include <cstddef>
#include <vector>

#include "wos/geometry/scene.hpp"
#include "wos/grid.hpp"
#include "wos/solver/start_point.hpp"

namespace welding {

inline std::size_t flat_index_2d(int i, int j, int ny) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(ny)
         + static_cast<std::size_t>(j);
}

inline bool is_boundary_index_2d(const wos::Grid &grid, int i, int j) {
    return i == 0 || i == grid.Nx - 1 || j == 0 || j == grid.Ny - 1;
}

inline std::vector<wos::solver::StartPoint<2>> build_start_points_2d(
        const wos::Grid &grid,
        const wos::GeometryScene<2> &geometry_scene) {
    std::vector<wos::solver::StartPoint<2>> starts(
        static_cast<std::size_t>(grid.Nx) * grid.Ny);
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            starts[flat_index_2d(i, j, grid.Ny)] =
                wos::solver::find_start_point(
                    geometry_scene, wos::Point2D{x, y});
        }
    }
    return starts;
}

} // namespace welding
