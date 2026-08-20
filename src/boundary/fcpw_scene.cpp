#include "wos/boundary/fcpw_scene.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <fcpw/fcpw.h>

namespace wos {

namespace {

struct CoordinateTransform2D {
    Point2D origin{};
    double scale = 1.0;

    fcpw::Vector2 point_to_fcpw(Point2D point) const {
        return fcpw::Vector2(
            checked_float((point.x - origin.x) / scale),
            checked_float((point.y - origin.y) / scale));
    }

    Point2D point_from_fcpw(const fcpw::Vector2 &point) const {
        return Point2D{
            origin.x + scale * static_cast<double>(point[0]),
            origin.y + scale * static_cast<double>(point[1]),
        };
    }

    float distance_to_fcpw(double distance) const {
        if (std::isinf(distance) && distance > 0.0) {
            return fcpw::maxFloat;
        }
        return checked_float(distance / scale);
    }

    double distance_from_fcpw(float distance) const {
        return scale * static_cast<double>(distance);
    }

private:
    static float checked_float(double value) {
        if (!std::isfinite(value) ||
            std::abs(value) > static_cast<double>(fcpw::maxFloat)) {
            throw std::range_error(
                "geometry value cannot be represented by FCPW");
        }
        return static_cast<float>(value);
    }
};

CoordinateTransform2D make_transform(const Mesh<2> &mesh) {
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (Point2D point : mesh.verts) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument(
                "FCPW boundary vertices must have finite coordinates");
        }
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }

    const double scale = std::max(xmax - xmin, ymax - ymin);
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::invalid_argument(
            "FCPW boundary must have a finite nonzero extent");
    }
    return CoordinateTransform2D{
        Point2D{0.5 * (xmin + xmax), 0.5 * (ymin + ymax)}, scale};
}

void validate_input(const Mesh<2> &mesh,
                    const std::vector<BoundaryType> &primitive_types) {
    if (mesh.n_prims() <= 0 || mesh.verts.empty()) {
        throw std::invalid_argument(
            "FcpwBoundaryScene2D requires at least one boundary segment");
    }
    if (primitive_types.size() !=
        static_cast<std::size_t>(mesh.n_prims())) {
        throw std::invalid_argument(
            "FcpwBoundaryScene2D requires one BoundaryType per primitive");
    }
    if (mesh.boundary_ids.size() !=
        static_cast<std::size_t>(mesh.n_prims())) {
        throw std::invalid_argument(
            "2D boundary mesh requires one boundary id per primitive");
    }
    for (int index : mesh.prims) {
        if (index < 0 || index >= mesh.n_verts()) {
            throw std::out_of_range(
                "boundary segment contains an invalid vertex index");
        }
    }
}

void append_primitive(const Mesh<2> &source, int primitive,
                      Mesh<2> &target, std::vector<int> &to_original) {
    target.prims.push_back(source.prims[2 * primitive]);
    target.prims.push_back(source.prims[2 * primitive + 1]);
    target.boundary_ids.push_back(source.boundary_ids[primitive]);
    to_original.push_back(primitive);
}

std::unique_ptr<fcpw::Scene<2>> build_scene(
    const Mesh<2> &mesh, const CoordinateTransform2D &transform,
    bool compute_silhouettes) {
    std::vector<fcpw::Vector2> positions;
    positions.reserve(mesh.verts.size());
    for (Point2D point : mesh.verts) {
        positions.push_back(transform.point_to_fcpw(point));
    }

    std::vector<fcpw::Vector2i> segments;
    segments.reserve(mesh.n_prims());
    for (int primitive = 0; primitive < mesh.n_prims(); ++primitive) {
        const int first = mesh.prims[2 * primitive];
        const int second = mesh.prims[2 * primitive + 1];
        if ((positions[first] - positions[second]).squaredNorm() == 0.0f) {
            throw std::invalid_argument(
                "boundary contains a segment that collapses at FCPW precision");
        }
        segments.emplace_back(first, second);
    }

    auto scene = std::make_unique<fcpw::Scene<2>>();
    scene->setObjectCount(1);
    scene->setObjectVertices(positions, 0);
    scene->setObjectLineSegments(segments, 0);
    if (compute_silhouettes) scene->computeSilhouettes();
    scene->build(fcpw::AggregateType::Bvh_SurfaceArea, false);
    return scene;
}

NearestPointResult<2> missing_nearest_result() {
    return NearestPointResult<2>{
        std::numeric_limits<double>::infinity(), Point2D{}, -1, -1};
}

Point2D outward_normal_from_mesh(const Mesh<2> &mesh,
                                 int primitive_id) {
    if (primitive_id < 0 || primitive_id >= mesh.n_prims()) {
        throw std::out_of_range("boundary primitive id is out of range");
    }
    const Point2D first = mesh.verts[mesh.prims[2 * primitive_id]];
    const Point2D second = mesh.verts[mesh.prims[2 * primitive_id + 1]];
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    const double length = std::hypot(dx, dy);
    if (!(length > 0.0)) {
        throw std::runtime_error(
            "cannot compute a normal for a zero-length segment");
    }
    return Point2D{dy / length, -dx / length};
}

} // namespace

struct FcpwBoundaryScene2D::Impl {
    Mesh<2> full_mesh;
    Mesh<2> dirichlet_mesh;
    Mesh<2> neumann_mesh;
    std::vector<int> dirichlet_to_original;
    std::vector<int> neumann_to_original;
    std::vector<int> neumann_vertex_to_original;
    CoordinateTransform2D transform;
    std::unique_ptr<fcpw::Scene<2>> full_scene;
    std::unique_ptr<fcpw::Scene<2>> dirichlet_scene;
    std::unique_ptr<fcpw::Scene<2>> neumann_scene;

    Impl(const Mesh<2> &mesh,
         const std::vector<BoundaryType> &primitive_types)
        : full_mesh(mesh), transform(make_transform(mesh)) {
        validate_input(mesh, primitive_types);
        dirichlet_mesh.verts = mesh.verts;
        neumann_mesh.verts = mesh.verts;
        neumann_vertex_to_original.assign(mesh.verts.size(), -1);

        for (int primitive = 0; primitive < mesh.n_prims(); ++primitive) {
            switch (primitive_types[primitive]) {
            case BoundaryType::Dirichlet:
                append_primitive(mesh, primitive, dirichlet_mesh,
                                 dirichlet_to_original);
                break;
            case BoundaryType::Neumann: {
                append_primitive(mesh, primitive, neumann_mesh,
                                 neumann_to_original);
                const int first = mesh.prims[2 * primitive];
                const int second = mesh.prims[2 * primitive + 1];
                if (neumann_vertex_to_original[first] < 0) {
                    neumann_vertex_to_original[first] = primitive;
                }
                if (neumann_vertex_to_original[second] < 0) {
                    neumann_vertex_to_original[second] = primitive;
                }
                break;
            }
            case BoundaryType::Robin:
                throw std::invalid_argument(
                    "FcpwBoundaryScene2D does not yet support Robin boundaries");
            }
        }

        full_scene = build_scene(full_mesh, transform, false);
        if (dirichlet_mesh.n_prims() > 0) {
            dirichlet_scene = build_scene(
                dirichlet_mesh, transform, false);
        }
        if (neumann_mesh.n_prims() > 0) {
            neumann_scene = build_scene(neumann_mesh, transform, true);
        }
    }

    NearestPointResult<2> closest(
        const fcpw::Scene<2> &scene, Point2D point,
        const std::vector<int> *to_original) const {
        fcpw::Interaction<2> interaction;
        if (!scene.findClosestPoint(
                transform.point_to_fcpw(point), interaction)) {
            return missing_nearest_result();
        }

        int primitive_id = interaction.primitiveIndex;
        if (to_original != nullptr) {
            primitive_id = to_original->at(
                static_cast<std::size_t>(primitive_id));
        }
        return NearestPointResult<2>{
            transform.distance_from_fcpw(interaction.d),
            transform.point_from_fcpw(interaction.p),
            primitive_id,
            full_mesh.boundary_ids.at(
                static_cast<std::size_t>(primitive_id)),
        };
    }

    RayHit2D intersect(
        const fcpw::Scene<2> &scene,
        const std::vector<int> *to_original, Point2D origin,
        Point2D direction, double minimum_distance,
        double maximum_distance) const {
        RayHit2D result;
        const double direction_length = std::hypot(direction.x, direction.y);
        if (!(direction_length > 0.0) ||
            !(maximum_distance >= minimum_distance)) {
            return result;
        }

        const Point2D unit{
            direction.x / direction_length,
            direction.y / direction_length,
        };
        const double normalized_minimum = std::max(0.0, minimum_distance);
        const Point2D shifted_origin{
            origin.x + normalized_minimum * unit.x,
            origin.y + normalized_minimum * unit.y,
        };
        const double remaining_distance =
            maximum_distance - normalized_minimum;
        if (!(remaining_distance >= 0.0)) return result;

        fcpw::Ray<2> ray(
            transform.point_to_fcpw(shifted_origin),
            fcpw::Vector2(
                static_cast<float>(unit.x),
                static_cast<float>(unit.y)),
            transform.distance_to_fcpw(remaining_distance));
        fcpw::Interaction<2> interaction;
        if (!scene.intersect(ray, interaction)) return result;

        int primitive_id = interaction.primitiveIndex;
        if (to_original != nullptr) {
            primitive_id = to_original->at(
                static_cast<std::size_t>(primitive_id));
        }
        result.hit = true;
        result.distance = normalized_minimum +
            transform.distance_from_fcpw(interaction.d);
        result.point = transform.point_from_fcpw(interaction.p);
        result.normal = Point2D{
            static_cast<double>(interaction.n[0]),
            static_cast<double>(interaction.n[1]),
        };
        result.primitive_id = primitive_id;
        result.boundary_id = full_mesh.boundary_ids.at(
            static_cast<std::size_t>(primitive_id));
        return result;
    }
};

FcpwBoundaryScene2D::FcpwBoundaryScene2D(
    const Mesh<2> &mesh,
    const std::vector<BoundaryType> &primitive_types)
    : impl_(std::make_unique<Impl>(mesh, primitive_types)) {}

FcpwBoundaryScene2D::~FcpwBoundaryScene2D() = default;
FcpwBoundaryScene2D::FcpwBoundaryScene2D(
    FcpwBoundaryScene2D &&) noexcept = default;
FcpwBoundaryScene2D &FcpwBoundaryScene2D::operator=(
    FcpwBoundaryScene2D &&) noexcept = default;

bool FcpwBoundaryScene2D::has_dirichlet() const {
    return impl_->dirichlet_scene != nullptr;
}

bool FcpwBoundaryScene2D::has_neumann() const {
    return impl_->neumann_scene != nullptr;
}

const Mesh<2> &FcpwBoundaryScene2D::full_mesh() const {
    return impl_->full_mesh;
}

const Mesh<2> &FcpwBoundaryScene2D::neumann_mesh() const {
    return impl_->neumann_mesh;
}

int FcpwBoundaryScene2D::original_neumann_primitive(
    int primitive_id) const {
    return impl_->neumann_to_original.at(
        static_cast<std::size_t>(primitive_id));
}

Point2D FcpwBoundaryScene2D::outward_normal(
    int original_primitive_id) const {
    return outward_normal_from_mesh(
        impl_->full_mesh, original_primitive_id);
}

NearestPointResult<2> FcpwBoundaryScene2D::closest_boundary(
    Point2D point) const {
    return impl_->closest(*impl_->full_scene, point, nullptr);
}

NearestPointResult<2> FcpwBoundaryScene2D::closest_dirichlet(
    Point2D point) const {
    if (!impl_->dirichlet_scene) return missing_nearest_result();
    return impl_->closest(
        *impl_->dirichlet_scene, point,
        &impl_->dirichlet_to_original);
}

NearestPointResult<2> FcpwBoundaryScene2D::closest_neumann(
    Point2D point) const {
    if (!impl_->neumann_scene) return missing_nearest_result();
    return impl_->closest(
        *impl_->neumann_scene, point,
        &impl_->neumann_to_original);
}

PointClassification FcpwBoundaryScene2D::classify_point(
    Point2D point, double boundary_distance,
    double boundary_tolerance, int max_ray_attempts,
    PRNG &rng) const {
    (void)max_ray_attempts;
    (void)rng;
    return classify_point_2d(
        impl_->full_mesh, point, boundary_distance,
        boundary_tolerance);
}

NeumannSilhouetteResult2D
FcpwBoundaryScene2D::closest_neumann_silhouette(Point2D point) const {
    NeumannSilhouetteResult2D result;
    if (!impl_->neumann_scene) return result;

    fcpw::Interaction<2> interaction;
    constexpr float precision =
        64.0f * std::numeric_limits<float>::epsilon();
    if (!impl_->neumann_scene->findClosestSilhouettePoint(
            impl_->transform.point_to_fcpw(point), interaction,
            false, 0.0f, fcpw::maxFloat, precision)) {
        return result;
    }

    const int vertex_id = interaction.primitiveIndex;
    if (vertex_id < 0 ||
        vertex_id >= static_cast<int>(
            impl_->neumann_vertex_to_original.size())) {
        return result;
    }
    const int primitive_id = impl_->neumann_vertex_to_original[vertex_id];
    if (primitive_id < 0) return result;

    result.found = true;
    result.distance = impl_->transform.distance_from_fcpw(interaction.d);
    result.point = impl_->transform.point_from_fcpw(interaction.p);
    result.primitive_id = primitive_id;
    result.boundary_id = impl_->full_mesh.boundary_ids.at(
        static_cast<std::size_t>(primitive_id));
    return result;
}

RayHit2D FcpwBoundaryScene2D::intersect_neumann(
    Point2D origin, Point2D direction, double minimum_distance,
    double maximum_distance) const {
    if (!impl_->neumann_scene) return RayHit2D{};
    return impl_->intersect(
        *impl_->neumann_scene, &impl_->neumann_to_original,
        origin, direction, minimum_distance, maximum_distance);
}

bool FcpwBoundaryScene2D::visible(
    Point2D first, Point2D second, double endpoint_tolerance) const {
    const Point2D direction{second.x - first.x, second.y - first.y};
    const double length = std::hypot(direction.x, direction.y);
    if (!(length > 0.0)) return true;

    const double float_tolerance =
        64.0 * std::numeric_limits<float>::epsilon() * impl_->transform.scale;
    const double endpoint_distance = std::max(
        endpoint_tolerance * length, float_tolerance);
    if (2.0 * endpoint_distance >= length) return true;
    return !impl_->intersect(
        *impl_->full_scene, nullptr, first, direction,
        endpoint_distance, length - endpoint_distance).hit;
}

} // namespace wos
