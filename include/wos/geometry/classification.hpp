#pragma once

#include "wos/mesh.hpp"

namespace wos {

enum class PointLocation {
    Outside,
    Inside,
    Boundary,
    Indeterminate,
};

struct PointClassification {
    PointLocation location;
    int ambiguous_ray_retries;
};

int winding_number(const Mesh<2> &mesh, Point2D point);

PointClassification classify_point_2d(
    const Mesh<2> &mesh, Point2D point, double boundary_distance,
    double boundary_tolerance);

} // namespace wos
