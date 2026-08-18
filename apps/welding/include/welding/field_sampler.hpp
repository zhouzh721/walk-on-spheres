#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "wos/grid.hpp"
#include "wos/mesh.hpp"

namespace welding {

// Read-only bilinear interpolation over a complete 2D Cartesian field.
// Values use the same (i, j) row-major order as wos::Field3D with Nz == 1.
class FieldSampler2D {
public:
    FieldSampler2D(const wos::Grid &grid, const std::vector<double> &values)
        : grid_(grid), values_(values) {
        const std::size_t expected =
            static_cast<std::size_t>(grid_.Nx) * grid_.Ny;
        if (grid_.Nx < 2 || grid_.Ny < 2 || values_.size() != expected) {
            throw std::invalid_argument(
                "FieldSampler2D requires a complete grid of at least 2 x 2");
        }
        if (!(grid_.xmax > grid_.xmin) || !(grid_.ymax > grid_.ymin)) {
            throw std::invalid_argument(
                "FieldSampler2D requires non-degenerate grid bounds");
        }
    }

    double sample(wos::Point2D point) const {
        const double gx = std::clamp(
            (point.x - grid_.xmin) / (grid_.xmax - grid_.xmin)
                * static_cast<double>(grid_.Nx - 1),
            0.0, static_cast<double>(grid_.Nx - 1));
        const double gy = std::clamp(
            (point.y - grid_.ymin) / (grid_.ymax - grid_.ymin)
                * static_cast<double>(grid_.Ny - 1),
            0.0, static_cast<double>(grid_.Ny - 1));

        const int i0 = static_cast<int>(std::floor(gx));
        const int j0 = static_cast<int>(std::floor(gy));
        const int i1 = std::min(i0 + 1, grid_.Nx - 1);
        const int j1 = std::min(j0 + 1, grid_.Ny - 1);
        const double tx = gx - static_cast<double>(i0);
        const double ty = gy - static_cast<double>(j0);

        const double v00 = at(i0, j0);
        const double v10 = at(i1, j0);
        const double v01 = at(i0, j1);
        const double v11 = at(i1, j1);

        const double lower = (1.0 - tx) * v00 + tx * v10;
        const double upper = (1.0 - tx) * v01 + tx * v11;
        return (1.0 - ty) * lower + ty * upper;
    }

private:
    double at(int i, int j) const {
        return values_[static_cast<std::size_t>(i) * grid_.Ny + j];
    }

    const wos::Grid &grid_;
    const std::vector<double> &values_;
};

} // namespace welding
