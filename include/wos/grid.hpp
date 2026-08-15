#pragma once

namespace wos {

struct Grid {
    int Nx, Ny, Nz;
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

inline constexpr double grid_coordinate(
        double min, double max, int index, int count) {
    return count > 1
        ? min + (max - min) * static_cast<double>(index) /
                static_cast<double>(count - 1)
        : min;
}

}
