#pragma once

#include <memory>
#include <vector>

#include "wos/boundary/scene.hpp"

namespace wos {

// FCPW-backed geometry queries. FCPW and Eigen types are hidden in Impl so
// the solver-facing API remains independent of the selected geometry library.
class FcpwBoundaryScene2D final : public BoundaryScene2D {
public:
    FcpwBoundaryScene2D(
        const Mesh<2> &mesh,
        const std::vector<BoundaryType> &primitive_types);
    ~FcpwBoundaryScene2D() override;

    FcpwBoundaryScene2D(const FcpwBoundaryScene2D &) = delete;
    FcpwBoundaryScene2D &operator=(const FcpwBoundaryScene2D &) = delete;
    FcpwBoundaryScene2D(FcpwBoundaryScene2D &&) noexcept;
    FcpwBoundaryScene2D &operator=(FcpwBoundaryScene2D &&) noexcept;

    bool has_dirichlet() const override;
    bool has_neumann() const override;

    const Mesh<2> &full_mesh() const override;
    const Mesh<2> &neumann_mesh() const override;

    int original_neumann_primitive(int primitive_id) const override;
    Point2D outward_normal(int original_primitive_id) const override;

    NearestPointResult<2> closest_boundary(Point2D point) const override;
    NearestPointResult<2> closest_dirichlet(Point2D point) const override;
    NearestPointResult<2> closest_neumann(Point2D point) const override;

    PointClassification classify_point(
        Point2D point, double boundary_distance,
        double boundary_tolerance, int max_ray_attempts,
        PRNG &rng) const override;

    NeumannSilhouetteResult2D closest_neumann_silhouette(
        Point2D point) const override;

    RayHit2D intersect_neumann(
        Point2D origin, Point2D direction, double minimum_distance,
        double maximum_distance) const override;

    bool visible(
        Point2D first, Point2D second,
        double endpoint_tolerance = 1e-10) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wos
