#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "wos/grid.hpp"
#include "wos/mesh.hpp"

namespace welding {

enum class FieldInterpolation2D {
    Bilinear,
    Cubic,
};

inline const char *field_interpolation_name(
    FieldInterpolation2D interpolation) {
    switch (interpolation) {
    case FieldInterpolation2D::Bilinear:
        return "bilinear";
    case FieldInterpolation2D::Cubic:
        return "cubic";
    }
    return "unknown";
}

// Read-only interpolation over a complete uniform 2D Cartesian field.
// Values use the same (i, j) row-major order as wos::Field3D with Nz == 1.
class FieldSampler2D {
public:
    FieldSampler2D(
        const wos::Grid &grid,
        const std::vector<double> &values,
        FieldInterpolation2D interpolation = FieldInterpolation2D::Bilinear)
        : grid_(grid), values_(values), interpolation_(interpolation) {
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

        if (interpolation_ == FieldInterpolation2D::Cubic &&
            grid_.Nx >= 4 && grid_.Ny >= 4) {
            return sample_cubic(gx, gy);
        }
        return sample_bilinear(gx, gy);
    }

private:
    double sample_bilinear(double gx, double gy) const {
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

    static void cubic_lagrange_weights(double coordinate, double weights[4]) {
        weights[0] = -(coordinate - 1.0) * (coordinate - 2.0)
                   * (coordinate - 3.0) / 6.0;
        weights[1] = coordinate * (coordinate - 2.0)
                   * (coordinate - 3.0) / 2.0;
        weights[2] = -coordinate * (coordinate - 1.0)
                   * (coordinate - 3.0) / 2.0;
        weights[3] = coordinate * (coordinate - 1.0)
                   * (coordinate - 2.0) / 6.0;
    }

    double sample_cubic(double gx, double gy) const {
        const int ix = std::clamp(
            static_cast<int>(std::floor(gx)) - 1, 0, grid_.Nx - 4);
        const int iy = std::clamp(
            static_cast<int>(std::floor(gy)) - 1, 0, grid_.Ny - 4);
        double wx[4];
        double wy[4];
        cubic_lagrange_weights(gx - static_cast<double>(ix), wx);
        cubic_lagrange_weights(gy - static_cast<double>(iy), wy);

        double value = 0.0;
        for (int di = 0; di < 4; ++di) {
            double row = 0.0;
            for (int dj = 0; dj < 4; ++dj) {
                row += wy[dj] * at(ix + di, iy + dj);
            }
            value += wx[di] * row;
        }
        return value;
    }

    double at(int i, int j) const {
        return values_[static_cast<std::size_t>(i) * grid_.Ny + j];
    }

    const wos::Grid &grid_;
    const std::vector<double> &values_;
    FieldInterpolation2D interpolation_;
};

} // namespace welding
