#include <algorithm>
#include <cmath>
#include <limits>
#include "wos/bvh.hpp"
#include "wos/inside.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"

namespace wos {

namespace {

constexpr int RAY_STACK_MAX = 64;
constexpr double INTERSECTION_EPSILON = 1e-9;
constexpr int REQUIRED_VALID_RAYS = 3;
constexpr int DEFAULT_MAX_RAY_ATTEMPTS = 12;

Vec3D random_unit_direction(PRNG &rng) {
    double u = rng.unit();
    double v = rng.unit();
    double z = 2*u-1;
    double phi = 2*M_PI*v;
    double r = std::sqrt(1-z*z);
    return Vec3D{r*std::cos(phi), r*std::sin(phi), z};
}

bool update_ray_interval(double origin, double direction,
                         double box_min, double box_max,
                         double &t_min, double &t_max) {
    // Expand the box slightly so round-off cannot incorrectly remove a
    // triangle that lies exactly on a box face.
    const double extent_scale = std::max(1.0, std::fabs(box_max - box_min));
    const double coordinate_scale =
        std::max({1.0, std::fabs(origin), std::fabs(box_min), std::fabs(box_max)});
    const double padding =
        INTERSECTION_EPSILON * extent_scale +
        16.0 * std::numeric_limits<double>::epsilon() * coordinate_scale;
    box_min -= padding;
    box_max += padding;

    if (direction == 0.0) {
        return origin >= box_min && origin <= box_max;
    }

    double t0 = (box_min - origin) / direction;
    double t1 = (box_max - origin) / direction;
    if (t0 > t1) std::swap(t0, t1);
    t_min = std::max(t_min, t0);
    t_max = std::min(t_max, t1);
    return t_min <= t_max;
}

bool ray_intersects_aabb(Point3D origin, Vec3D ray, const AABB<3> &box) {
    double t_min = 0.0;
    double t_max = std::numeric_limits<double>::infinity();
    return update_ray_interval(origin.x, ray.x, box.pmin.x, box.pmax.x,
                               t_min, t_max) &&
           update_ray_interval(origin.y, ray.y, box.pmin.y, box.pmax.y,
                               t_min, t_max) &&
           update_ray_interval(origin.z, ray.z, box.pmin.z, box.pmax.z,
                               t_min, t_max);
}

RayCastResult count_ray_tri_intersect(const Mesh<3> &m, Point3D p, Vec3D ray) {
    int intersects = 0;
    for (int s = 0; s < m.n_prims(); s++) {
        Point3D v0 = m.verts[m.prims[3*s + 0]];
        Point3D v1 = m.verts[m.prims[3*s + 1]];
        Point3D v2 = m.verts[m.prims[3*s + 2]];

        const RayHit hit = moller_trumbore(p, ray, v0, v1, v2);
        if (hit == RayHit::Ambiguous) return RayCastResult{intersects, true};
        if (hit == RayHit::Hit) ++intersects;
    }
    return RayCastResult{intersects, false};
}

RayCastResult count_ray_tri_intersect(const BVH<3> &bvh, Point3D p, Vec3D ray) {
    const Mesh<3> &m = *bvh.mesh;
    int intersects = 0;

    int stack[RAY_STACK_MAX];
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const int node_idx = stack[--stack_size];
        const BVHNode<3> &node = bvh.nodes[node_idx];
        if (!ray_intersects_aabb(p, ray, node.bbox)) continue;

        if (node.prim_count > 0) {
            for (int i = 0; i < node.prim_count; ++i) {
                const int tri = bvh.prims[node.idx + i];
                Point3D v0 = m.verts[m.prims[3*tri + 0]];
                Point3D v1 = m.verts[m.prims[3*tri + 1]];
                Point3D v2 = m.verts[m.prims[3*tri + 2]];
                const RayHit hit = moller_trumbore(p, ray, v0, v1, v2);
                if (hit == RayHit::Ambiguous) {
                    return RayCastResult{intersects, true};
                }
                if (hit == RayHit::Hit) ++intersects;
            }
        } else {
            const int left = node_idx + 1;
            const int right = node.idx;
            stack[stack_size++] = right;
            stack[stack_size++] = left;
        }
    }
    return RayCastResult{intersects, false};
}

}

// check which side of a line segment a point p is on using signed triangle area
// > 0 : left, < 0 : right, = 0 : collinear
static inline double signed_tri_area(Point2D s0, Point2D s1, Point2D p) {
    return (s1.x - s0.x)*(p.y - s0.y) - (p.x - s0.x)*(s1.y - s0.y);
}

// Sunday's winding number algorithm for point-in-polygon tests (2D)
int winding_number(const Mesh<2> &m, Point2D p) {
    int wn = 0;

    for (int s = 0; s < m.n_prims(); s++) {
        Point2D s0 = m.verts[m.prims[2*s + 0]];
        Point2D s1 = m.verts[m.prims[2*s + 1]];

        if (s0.y <= p.y) {
            if (s1.y > p.y && signed_tri_area(s0, s1, p) > 0) wn++;
        } else {
            if (s1.y <= p.y && signed_tri_area(s0, s1, p) < 0) wn--;
        }
    }

    return wn;
}

// Möller-Trumbore ray-triangle intersection algorithm
RayHit moller_trumbore(Point3D p, Vec3D ray, Point3D a, Point3D b, Point3D c) {
    Vec3D edge1 = Vec3D{b.x - a.x, b.y - a.y, b.z - a.z};
    Vec3D edge2 = Vec3D{c.x - a.x, c.y - a.y, c.z - a.z};
    Vec3D s = Vec3D{p.x - a.x, p.y - a.y, p.z - a.z};

    Vec3D rxe2 = cross(ray, edge2);
    double det = dot(edge1, rxe2);
    Vec3D normal = cross(edge1, edge2);
    const double normal_length = std::sqrt(dot(normal, normal));
    const double det_tolerance = INTERSECTION_EPSILON * normal_length;
    if (std::fabs(det) <= det_tolerance) {
        if (normal_length == 0.0) return RayHit::Miss;
        const double edge_scale = std::max({
            1.0,
            std::sqrt(dot(edge1, edge1)),
            std::sqrt(dot(edge2, edge2)),
        });
        const double plane_distance = std::fabs(dot(normal, s)) / normal_length;
        return plane_distance <= INTERSECTION_EPSILON * edge_scale
            ? RayHit::Ambiguous
            : RayHit::Miss;
    }

    double u = dot(s, rxe2) / det;
    if (u < -INTERSECTION_EPSILON ||
        u - 1.0 > INTERSECTION_EPSILON) return RayHit::Miss;

    Vec3D sxe1 = cross(s, edge1);
    double v = dot(ray, sxe1) / det;
    if (v < -INTERSECTION_EPSILON ||
        u + v - 1.0 > INTERSECTION_EPSILON) return RayHit::Miss;

    const double t = dot(edge2, sxe1) / det;
    if (t <= INTERSECTION_EPSILON) return RayHit::Miss;

    const double w = 1.0 - u - v;
    if (u <= INTERSECTION_EPSILON ||
        v <= INTERSECTION_EPSILON ||
        w <= INTERSECTION_EPSILON) {
        return RayHit::Ambiguous;
    }
    return RayHit::Hit;
}

// aggregate # ray-triangle intersections over entire Mesh (3D)
RayCastResult count_ray_tri_intersect(const Mesh<3> &m, Point3D p, PRNG &rng) {
    const Vec3D ray = random_unit_direction(rng);
    // TODO - test for edge cases on intersection of multiple faces (potentially over/under-counted), re-randomise ray
    return count_ray_tri_intersect(m, p, ray);
}

RayCastResult count_ray_tri_intersect(const BVH<3> &bvh, Point3D p, PRNG &rng) {
    const Vec3D ray = random_unit_direction(rng);
    return count_ray_tri_intersect(bvh, p, ray);
}

PointLocation classify_point(const Mesh<2> &m, Point2D p,
                             double boundary_distance, double boundary_tolerance) {
    if (boundary_distance <= boundary_tolerance) return PointLocation::Boundary;
    return winding_number(m, p) != 0 ? PointLocation::Inside : PointLocation::Outside;
}

PointLocation classify_point(const Mesh<3> &m, Point3D p,
                             double boundary_distance, double boundary_tolerance, PRNG &rng) {
    if (boundary_distance <= boundary_tolerance) return PointLocation::Boundary;

    int valid_rays = 0;
    int inside_votes = 0;
    for (int attempts = 0;
         attempts < DEFAULT_MAX_RAY_ATTEMPTS && valid_rays < REQUIRED_VALID_RAYS;
         ++attempts) {
        const RayCastResult cast = count_ray_tri_intersect(m, p, rng);
        if (cast.ambiguous) continue;
        inside_votes += (cast.intersections & 1);
        ++valid_rays;
    }
    if (valid_rays < REQUIRED_VALID_RAYS) return PointLocation::Indeterminate;
    return inside_votes > REQUIRED_VALID_RAYS/2
        ? PointLocation::Inside
        : PointLocation::Outside;
}

PointLocation classify_point(const BVH<3> &bvh, Point3D p,
                             double boundary_distance, double boundary_tolerance, PRNG &rng) {
    return classify_point(bvh, p, boundary_distance, boundary_tolerance,
                          DEFAULT_MAX_RAY_ATTEMPTS, rng).location;
}

PointClassification classify_point(const BVH<3> &bvh, Point3D p,
                                   double boundary_distance, double boundary_tolerance,
                                   int max_ray_attempts, PRNG &rng) {
    if (boundary_distance <= boundary_tolerance) {
        return PointClassification{PointLocation::Boundary, 0};
    }

    int valid_rays = 0;
    int inside_votes = 0;
    int ambiguous_ray_retries = 0;
    for (int attempts = 0;
         attempts < max_ray_attempts && valid_rays < REQUIRED_VALID_RAYS;
         ++attempts) {
        const RayCastResult cast = count_ray_tri_intersect(bvh, p, rng);
        if (cast.ambiguous) {
            ++ambiguous_ray_retries;
            continue;
        }
        inside_votes += (cast.intersections & 1);
        ++valid_rays;
    }
    if (valid_rays < REQUIRED_VALID_RAYS) {
        return PointClassification{
            PointLocation::Indeterminate, ambiguous_ray_retries
        };
    }
    return PointClassification{
        inside_votes > REQUIRED_VALID_RAYS/2
            ? PointLocation::Inside
            : PointLocation::Outside,
        ambiguous_ray_retries
    };
}

}
