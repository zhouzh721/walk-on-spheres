#include "wos/sampling/neumann_boundary.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wos {

namespace {

struct SegmentSpan {
    int primitive;
    double first_parameter;
    double last_parameter;
    double length;
};

bool segment_span_in_disk(Point2D first, Point2D second, Point2D center,
                          double radius, double &first_parameter,
                          double &last_parameter) {
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    const double ox = first.x - center.x;
    const double oy = first.y - center.y;
    const double a = dx * dx + dy * dy;
    if (!(a > 0.0) || !(radius > 0.0)) return false;

    const double b = 2.0 * (ox * dx + oy * dy);
    const double c = ox * ox + oy * oy - radius * radius;
    const double discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        if (c > 0.0) return false;
        first_parameter = 0.0;
        last_parameter = 1.0;
        return true;
    }

    const double root = std::sqrt(std::max(0.0, discriminant));
    const double enter = (-b - root) / (2.0 * a);
    const double leave = (-b + root) / (2.0 * a);
    first_parameter = std::max(0.0, enter);
    last_parameter = std::min(1.0, leave);
    return last_parameter > first_parameter;
}

} // namespace

NeumannBoundarySample2D sample_neumann_boundary(
    const BoundaryScene2D &scene, Point2D center, double radius, PRNG &rng) {
    NeumannBoundarySample2D result;
    if (!scene.has_neumann() || !(radius > 0.0)) return result;

    const Mesh<2> &mesh = scene.neumann_mesh();
    std::vector<SegmentSpan> spans;
    spans.reserve(mesh.n_prims());
    double total_length = 0.0;

    for (int primitive = 0; primitive < mesh.n_prims(); ++primitive) {
        const Point2D first = mesh.verts[mesh.prims[2 * primitive]];
        const Point2D second = mesh.verts[mesh.prims[2 * primitive + 1]];
        double first_parameter = 0.0;
        double last_parameter = 0.0;
        if (!segment_span_in_disk(first, second, center, radius,
                                  first_parameter, last_parameter)) {
            continue;
        }

        const double full_length = dist(first, second);
        const double span_length =
            (last_parameter - first_parameter) * full_length;
        if (!(span_length > 0.0)) continue;
        spans.push_back(SegmentSpan{
            primitive, first_parameter, last_parameter, span_length});
        total_length += span_length;
    }

    if (!(total_length > 0.0)) return result;

    double target = rng.unit() * total_length;
    const SegmentSpan *selected = &spans.back();
    for (const SegmentSpan &span : spans) {
        if (target <= span.length) {
            selected = &span;
            break;
        }
        target -= span.length;
    }

    const Point2D first =
        mesh.verts[mesh.prims[2 * selected->primitive]];
    const Point2D second =
        mesh.verts[mesh.prims[2 * selected->primitive + 1]];
    const double parameter = selected->first_parameter
        + (selected->last_parameter - selected->first_parameter)
            * rng.unit();
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    const double length = std::hypot(dx, dy);

    result.valid = true;
    result.point = Point2D{
        first.x + parameter * dx,
        first.y + parameter * dy,
    };
    result.normal = Point2D{dy / length, -dx / length};
    result.primitive_id =
        scene.original_neumann_primitive(selected->primitive);
    result.boundary_id = mesh.boundary_ids[selected->primitive];
    result.pdf = 1.0 / total_length;
    return result;
}

} // namespace wos
