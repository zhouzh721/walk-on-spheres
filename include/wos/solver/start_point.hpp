#pragma once

#include "wos/bvh.hpp"

namespace wos::solver {

template<int N>
struct StartPoint {
    double radius;
    Point<N> nearest;
    int boundary_id;
};

template<int N>
StartPoint<N> find_start_point(const BVH<N> &bvh, Point<N> point) {
    if constexpr (N == 2) {
        const auto nearest = bvh_npq(bvh, point);
        return StartPoint<N>{
            nearest.distance, nearest.point, nearest.boundary_id};
    } else {
        Point<N> nearest;
        const double radius = bvh_npq(bvh, point, &nearest);
        return StartPoint<N>{radius, nearest, -1};
    }
}

} // namespace wos::solver
