#pragma once

#include <memory>
#include <vector>

#include "wos/boundary/type.hpp"
#include "wos/geometry/neumann_silhouette.hpp"
#include "wos/geometry/ray_intersection.hpp"
#include "wos/geometry/scene.hpp"

namespace wos {

// Geometry-query interface used by the 2D WoSt solver. Implementations own
// the full boundary and its Dirichlet/Neumann subsets, while primitive ids in
// all returned results refer to the original full mesh.
class BoundaryScene2D : public GeometryScene<2> {
public:
    virtual ~BoundaryScene2D() = default;

    virtual bool has_dirichlet() const = 0;
    virtual bool has_neumann() const = 0;

    virtual const Mesh<2> &full_mesh() const = 0;
    const Mesh<2> &mesh() const final { return full_mesh(); }
    virtual const Mesh<2> &neumann_mesh() const = 0;

    virtual int original_neumann_primitive(int primitive_id) const = 0;
    virtual Point2D outward_normal(int original_primitive_id) const = 0;

    virtual NearestPointResult<2> closest_boundary(
        Point2D point) const override = 0;
    virtual NearestPointResult<2> closest_dirichlet(Point2D point) const = 0;
    virtual NearestPointResult<2> closest_neumann(Point2D point) const = 0;

    virtual NeumannSilhouetteResult2D closest_neumann_silhouette(
        Point2D point) const = 0;

    virtual RayHit2D intersect_neumann(
        Point2D origin, Point2D direction, double minimum_distance,
        double maximum_distance) const = 0;

    virtual bool visible(
        Point2D first, Point2D second,
        double endpoint_tolerance = 1e-10) const = 0;
};

std::unique_ptr<BoundaryScene2D> make_boundary_scene_2d(
    const Mesh<2> &mesh,
    const std::vector<BoundaryType> &primitive_types);

template<typename Equation>
std::vector<BoundaryType> classify_boundary_primitives(
    const Mesh<2> &mesh, const Equation &equation) {
    std::vector<BoundaryType> types;
    types.reserve(mesh.n_prims());
    for (int primitive = 0; primitive < mesh.n_prims(); ++primitive) {
        types.push_back(
            equation.boundary_type(mesh.boundary_ids[primitive]));
    }
    return types;
}

} // namespace wos
