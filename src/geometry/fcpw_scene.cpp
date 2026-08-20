#include "wos/geometry/fcpw_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fcpw/fcpw.h>

namespace wos {

namespace {

constexpr int required_valid_rays = 3;

double coordinate(Point2D point, int dimension) {
    return dimension == 0 ? point.x : point.y;
}

double coordinate(Point3D point, int dimension) {
    if (dimension == 0) return point.x;
    if (dimension == 1) return point.y;
    return point.z;
}

template<int N>
Point<N> make_point(const std::array<double, N> &coordinates) {
    if constexpr (N == 2) {
        return Point2D{coordinates[0], coordinates[1]};
    } else {
        return Point3D{
            coordinates[0], coordinates[1], coordinates[2]};
    }
}

float checked_float(double value) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(fcpw::maxFloat)) {
        throw std::range_error(
            "geometry value cannot be represented by FCPW");
    }
    return static_cast<float>(value);
}

template<int N>
struct CoordinateTransform {
    Point<N> origin{};
    double scale = 1.0;

    fcpw::Vector<N> point_to_fcpw(Point<N> point) const {
        fcpw::Vector<N> result;
        for (int dimension = 0; dimension < N; ++dimension) {
            result[dimension] = checked_float(
                (coordinate(point, dimension) -
                 coordinate(origin, dimension)) / scale);
        }
        return result;
    }

    Point<N> point_from_fcpw(const fcpw::Vector<N> &point) const {
        std::array<double, N> result{};
        for (int dimension = 0; dimension < N; ++dimension) {
            result[dimension] = coordinate(origin, dimension) +
                scale * static_cast<double>(point[dimension]);
        }
        return make_point<N>(result);
    }

    double distance_from_fcpw(float distance) const {
        return scale * static_cast<double>(distance);
    }
};

template<int N>
CoordinateTransform<N> make_transform(const Mesh<N> &mesh) {
    if (mesh.verts.empty()) {
        throw std::invalid_argument(
            "FCPW geometry requires at least one vertex");
    }

    std::array<double, N> minimum;
    std::array<double, N> maximum;
    minimum.fill(std::numeric_limits<double>::infinity());
    maximum.fill(-std::numeric_limits<double>::infinity());
    for (Point<N> point : mesh.verts) {
        for (int dimension = 0; dimension < N; ++dimension) {
            const double value = coordinate(point, dimension);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "FCPW geometry vertices must have finite coordinates");
            }
            minimum[dimension] = std::min(minimum[dimension], value);
            maximum[dimension] = std::max(maximum[dimension], value);
        }
    }

    std::array<double, N> center{};
    double scale = 0.0;
    for (int dimension = 0; dimension < N; ++dimension) {
        center[dimension] =
            0.5 * (minimum[dimension] + maximum[dimension]);
        scale = std::max(scale, maximum[dimension] - minimum[dimension]);
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::invalid_argument(
            "FCPW geometry must have a finite nonzero extent");
    }
    return CoordinateTransform<N>{make_point<N>(center), scale};
}

template<int N>
void validate_mesh(const Mesh<N> &mesh) {
    if (mesh.n_prims() <= 0 ||
        mesh.prims.size() % static_cast<std::size_t>(N) != 0) {
        throw std::invalid_argument(
            "FCPW geometry requires at least one complete primitive");
    }
    for (int index : mesh.prims) {
        if (index < 0 || index >= mesh.n_verts()) {
            throw std::out_of_range(
                "geometry primitive contains an invalid vertex index");
        }
    }
    if constexpr (N == 2) {
        if (mesh.boundary_ids.size() !=
            static_cast<std::size_t>(mesh.n_prims())) {
            throw std::invalid_argument(
                "2D geometry requires one boundary id per segment");
        }
    }
}

void validate_closed_manifold(const Mesh<3> &mesh) {
    std::map<std::pair<int, int>, int> edge_counts;
    for (int triangle = 0; triangle < mesh.n_prims(); ++triangle) {
        for (int edge = 0; edge < 3; ++edge) {
            int first = mesh.prims[3 * triangle + edge];
            int second = mesh.prims[3 * triangle + (edge + 1) % 3];
            if (first > second) std::swap(first, second);
            ++edge_counts[{first, second}];
        }
    }
    for (const auto &[edge, count] : edge_counts) {
        (void)edge;
        if (count != 2) {
            throw std::invalid_argument(
                "3D solver geometry must be a closed two-manifold mesh");
        }
    }
}

Point3D random_unit_direction(PRNG &rng) {
    const double z = 2.0 * rng.unit() - 1.0;
    const double angle = 2.0 * M_PI * rng.unit();
    const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    return Point3D{
        radius * std::cos(angle), radius * std::sin(angle), z};
}

struct RayIntersectionCount {
    int intersections = 0;
    bool ambiguous = false;
};

} // namespace

template<int N>
struct FcpwGeometryScene<N>::Impl {
    Mesh<N> mesh_data;
    CoordinateTransform<N> transform;
    fcpw::Scene<N> scene;

    explicit Impl(const Mesh<N> &mesh)
        : mesh_data(mesh), transform(make_transform(mesh)) {
        validate_mesh(mesh_data);
        if constexpr (N == 3) validate_closed_manifold(mesh_data);

        std::vector<fcpw::Vector<N>> positions;
        positions.reserve(mesh_data.verts.size());
        for (Point<N> point : mesh_data.verts) {
            positions.push_back(transform.point_to_fcpw(point));
        }

        scene.setObjectCount(1);
        scene.setObjectVertices(positions, 0);
        if constexpr (N == 2) {
            std::vector<fcpw::Vector2i> segments;
            segments.reserve(mesh_data.n_prims());
            for (int primitive = 0;
                 primitive < mesh_data.n_prims(); ++primitive) {
                const int first = mesh_data.prims[2 * primitive];
                const int second = mesh_data.prims[2 * primitive + 1];
                if ((positions[first] - positions[second]).squaredNorm() ==
                    0.0f) {
                    throw std::invalid_argument(
                        "segment collapses at FCPW precision");
                }
                segments.emplace_back(first, second);
            }
            scene.setObjectLineSegments(segments, 0);
        } else {
            std::vector<fcpw::Vector3i> triangles;
            triangles.reserve(mesh_data.n_prims());
            for (int primitive = 0;
                 primitive < mesh_data.n_prims(); ++primitive) {
                const int first = mesh_data.prims[3 * primitive];
                const int second = mesh_data.prims[3 * primitive + 1];
                const int third = mesh_data.prims[3 * primitive + 2];
                const auto first_edge = positions[second] - positions[first];
                const auto second_edge = positions[third] - positions[first];
                if (first_edge.cross(second_edge).squaredNorm() == 0.0f) {
                    throw std::invalid_argument(
                        "triangle collapses at FCPW precision");
                }
                triangles.emplace_back(first, second, third);
            }
            scene.setObjectTriangles(triangles, 0);
        }
        scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false);
    }

    NearestPointResult<N> closest(Point<N> point) const {
        fcpw::Interaction<N> interaction;
        if (!scene.findClosestPoint(
                transform.point_to_fcpw(point), interaction)) {
            return NearestPointResult<N>{
                std::numeric_limits<double>::infinity(), Point<N>{}, -1, -1};
        }
        int boundary_id = -1;
        if constexpr (N == 2) {
            boundary_id = mesh_data.boundary_ids.at(
                static_cast<std::size_t>(interaction.primitiveIndex));
        }
        return NearestPointResult<N>{
            transform.distance_from_fcpw(interaction.d),
            transform.point_from_fcpw(interaction.p),
            interaction.primitiveIndex,
            boundary_id,
        };
    }

    RayIntersectionCount count_intersections(
        Point<N> point, Point<N> direction) const {
        if constexpr (N == 2) {
            (void)point;
            (void)direction;
            return RayIntersectionCount{};
        } else {
            fcpw::Vector3 current = transform.point_to_fcpw(point);
            fcpw::Vector3 ray_direction(
                checked_float(direction.x), checked_float(direction.y),
                checked_float(direction.z));
            ray_direction.normalize();

            constexpr float barycentric_tolerance =
                64.0f * std::numeric_limits<float>::epsilon();
            constexpr float advance =
                32.0f * std::numeric_limits<float>::epsilon();
            RayIntersectionCount result;
            for (int hit_index = 0;
                 hit_index <= mesh_data.n_prims(); ++hit_index) {
                fcpw::Ray<3> ray(current, ray_direction);
                fcpw::Interaction<3> interaction;
                if (!scene.intersectRobust(ray, interaction)) return result;

                const float first = interaction.uv[0];
                const float second = interaction.uv[1];
                const float third = 1.0f - first - second;
                if (first <= barycentric_tolerance ||
                    second <= barycentric_tolerance ||
                    third <= barycentric_tolerance) {
                    result.ambiguous = true;
                    return result;
                }

                ++result.intersections;
                current = interaction.p + advance * ray_direction;
            }

            result.ambiguous = true;
            return result;
        }
    }

    PointClassification classify(
        Point<N> point, double boundary_distance,
        double boundary_tolerance, int max_ray_attempts,
        PRNG &rng) const {
        if constexpr (N == 2) {
            (void)max_ray_attempts;
            (void)rng;
            return classify_point_2d(
                mesh_data, point, boundary_distance,
                boundary_tolerance);
        } else {
            if (boundary_distance <= boundary_tolerance) {
                return PointClassification{PointLocation::Boundary, 0};
            }

            int valid_rays = 0;
            int inside_votes = 0;
            int ambiguous_ray_retries = 0;
            for (int attempt = 0;
                 attempt < max_ray_attempts &&
                 valid_rays < required_valid_rays;
                 ++attempt) {
                const RayIntersectionCount cast = count_intersections(
                    point, random_unit_direction(rng));
                if (cast.ambiguous) {
                    ++ambiguous_ray_retries;
                    continue;
                }
                inside_votes += cast.intersections & 1;
                ++valid_rays;
            }
            if (valid_rays < required_valid_rays) {
                return PointClassification{
                    PointLocation::Indeterminate,
                    ambiguous_ray_retries,
                };
            }
            return PointClassification{
                inside_votes > required_valid_rays / 2
                    ? PointLocation::Inside
                    : PointLocation::Outside,
                ambiguous_ray_retries,
            };
        }
    }
};

template<int N>
FcpwGeometryScene<N>::FcpwGeometryScene(const Mesh<N> &mesh)
    : impl_(std::make_unique<Impl>(mesh)) {}

template<int N>
FcpwGeometryScene<N>::~FcpwGeometryScene() = default;

template<int N>
FcpwGeometryScene<N>::FcpwGeometryScene(
    FcpwGeometryScene &&) noexcept = default;

template<int N>
FcpwGeometryScene<N> &FcpwGeometryScene<N>::operator=(
    FcpwGeometryScene &&) noexcept = default;

template<int N>
const Mesh<N> &FcpwGeometryScene<N>::mesh() const {
    return impl_->mesh_data;
}

template<int N>
NearestPointResult<N> FcpwGeometryScene<N>::closest_boundary(
    Point<N> point) const {
    return impl_->closest(point);
}

template<int N>
PointClassification FcpwGeometryScene<N>::classify_point(
    Point<N> point, double boundary_distance,
    double boundary_tolerance, int max_ray_attempts,
    PRNG &rng) const {
    return impl_->classify(
        point, boundary_distance, boundary_tolerance,
        max_ray_attempts, rng);
}

template class FcpwGeometryScene<2>;
template class FcpwGeometryScene<3>;

} // namespace wos
