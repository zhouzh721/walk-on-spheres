#pragma once

#include "wos/geometry/scene.hpp"

namespace wos::solver {

template<int N>
struct StartPoint {
    double radius;
    Point<N> nearest;
    int boundary_id;
};

template<int N>
StartPoint<N> find_start_point(
    const GeometryScene<N> &scene, Point<N> point) {
    const NearestPointResult<N> nearest =
        scene.closest_boundary(point);
    return StartPoint<N>{
        nearest.distance, nearest.point, nearest.boundary_id};
}

} // namespace wos::solver
