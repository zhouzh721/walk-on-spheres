#pragma once

#include <stdexcept>

#include "wos/mesh.hpp"

namespace welding {

enum class RectangularBoundary : int {
    Bottom = 0,
    Right = 1,
    Top = 2,
    Left = 3,
};

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
    mesh.boundary_ids = {
        static_cast<int>(RectangularBoundary::Bottom),
        static_cast<int>(RectangularBoundary::Right),
        static_cast<int>(RectangularBoundary::Top),
        static_cast<int>(RectangularBoundary::Left),
    };
    return mesh;
}

} // namespace welding
