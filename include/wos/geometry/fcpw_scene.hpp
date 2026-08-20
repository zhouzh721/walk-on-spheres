#pragma once

#include <memory>

#include "wos/geometry/scene.hpp"

namespace wos {

template<int N>
class FcpwGeometryScene final : public GeometryScene<N> {
public:
    explicit FcpwGeometryScene(const Mesh<N> &mesh);
    ~FcpwGeometryScene() override;

    FcpwGeometryScene(const FcpwGeometryScene &) = delete;
    FcpwGeometryScene &operator=(const FcpwGeometryScene &) = delete;
    FcpwGeometryScene(FcpwGeometryScene &&) noexcept;
    FcpwGeometryScene &operator=(FcpwGeometryScene &&) noexcept;

    const Mesh<N> &mesh() const override;
    NearestPointResult<N> closest_boundary(
        Point<N> point) const override;

    PointClassification classify_point(
        Point<N> point, double boundary_distance,
        double boundary_tolerance, int max_ray_attempts,
        PRNG &rng) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class FcpwGeometryScene<2>;
extern template class FcpwGeometryScene<3>;

} // namespace wos
