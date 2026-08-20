#include "wos/geometry/classification.hpp"

namespace wos {

namespace {

double signed_triangle_area(Point2D first, Point2D second, Point2D point) {
    return (second.x - first.x) * (point.y - first.y) -
        (point.x - first.x) * (second.y - first.y);
}

} // namespace

int winding_number(const Mesh<2> &mesh, Point2D point) {
    int winding = 0;
    for (int primitive = 0; primitive < mesh.n_prims(); ++primitive) {
        const Point2D first =
            mesh.verts[mesh.prims[2 * primitive]];
        const Point2D second =
            mesh.verts[mesh.prims[2 * primitive + 1]];

        if (first.y <= point.y) {
            if (second.y > point.y &&
                signed_triangle_area(first, second, point) > 0.0) {
                ++winding;
            }
        } else if (second.y <= point.y &&
                   signed_triangle_area(first, second, point) < 0.0) {
            --winding;
        }
    }
    return winding;
}

PointClassification classify_point_2d(
    const Mesh<2> &mesh, Point2D point, double boundary_distance,
    double boundary_tolerance) {
    if (boundary_distance <= boundary_tolerance) {
        return PointClassification{PointLocation::Boundary, 0};
    }
    return PointClassification{
        winding_number(mesh, point) != 0
            ? PointLocation::Inside
            : PointLocation::Outside,
        0,
    };
}

} // namespace wos
