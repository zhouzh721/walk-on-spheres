#pragma once
#include "wos/bvh.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"

namespace wos {

enum class PointLocation {
    Outside,
    Inside,
    Boundary,
    Indeterminate,
};

enum class RayHit {
    Miss,
    Hit,
    Ambiguous,
};

struct RayCastResult {
    int intersections;
    bool ambiguous;
};

struct PointClassification {
    PointLocation location;
    int ambiguous_ray_retries;
};

int winding_number(const Mesh<2> &m, Point2D p);
RayHit moller_trumbore(Point3D p, Vec3D ray, Point3D a, Point3D b, Point3D c);
RayCastResult count_ray_tri_intersect(const Mesh<3> &m, Point3D p, PRNG &rng);
RayCastResult count_ray_tri_intersect(const BVH<3> &bvh, Point3D p, PRNG &rng);

PointLocation classify_point(const Mesh<2> &m, Point2D p,
                             double boundary_distance, double boundary_tolerance);
PointLocation classify_point(const Mesh<3> &m, Point3D p,
                             double boundary_distance, double boundary_tolerance, PRNG &rng);
PointLocation classify_point(const BVH<3> &bvh, Point3D p,
                             double boundary_distance, double boundary_tolerance, PRNG &rng);
PointClassification classify_point(const BVH<3> &bvh, Point3D p,
                                   double boundary_distance, double boundary_tolerance,
                                   int max_ray_attempts, PRNG &rng);

}
