#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "wos/boundary/condition.hpp"
#include "wos/boundary/scene.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/prng.hpp"
#include "wos/sampling/neumann_boundary.hpp"
#include "wos/sampling/source_mis.hpp"
#include "wos/sampling/uniform.hpp"
#include "wos/solver/result.hpp"
#include "wos/solver/settings.hpp"
#include "wos/solver/wost_state.hpp"
#include "wos/source_mode.hpp"

namespace wos::solver {

namespace wost_detail {

struct PathResult {
    double value;
    int steps;
    TerminationReason termination;
};

inline double dot_2d(Point2D first, Point2D second) {
    return first.x * second.x + first.y * second.y;
}

inline Point2D sample_walk_direction(PRNG &rng, bool on_neumann,
                                     Point2D outward_normal) {
    const double angle = 2.0 * M_PI * rng.unit();
    Point2D direction{std::cos(angle), std::sin(angle)};
    if (on_neumann) {
        const double outward_component = dot_2d(direction, outward_normal);
        if (outward_component > 0.0) {
            direction.x -= 2.0 * outward_component * outward_normal.x;
            direction.y -= 2.0 * outward_component * outward_normal.y;
        }
    }
    return direction;
}

template<typename Equation>
double source_contribution(const BoundaryScene2D &scene,
                           const Equation &equation,
                           const Sphere2D &sphere, double throughput,
                           PRNG &source_rng) {
    if constexpr (!Equation::has_source) {
        (void)scene;
        (void)equation;
        (void)sphere;
        (void)throughput;
        (void)source_rng;
        return 0.0;
    } else {
        if (!equation.source_may_intersect(sphere)) return 0.0;

        if constexpr (Equation::has_green_source) {
            if (equation.source_mode == SourceMode::Green) {
                const Point2D source_point =
                    equation.sample_green(sphere, source_rng);
                if (!scene.visible(sphere.centre, source_point)) return 0.0;
                return throughput * equation.green_mass(sphere.radius)
                    * equation.source(source_point);
            }
            if (equation.source_mode == SourceMode::GreenSourceMIS) {
                if (scene.has_neumann()) {
                    throw std::invalid_argument(
                        "GreenSourceMIS is not yet supported by WoSt with Neumann boundaries");
                }
                if constexpr (source_mis::has_source_proposal_v<
                                  Equation, Point2D>) {
                    return throughput
                        * source_mis::green_source_mis_contribution(
                            equation, sphere, source_rng);
                } else {
                    throw std::invalid_argument(
                        "GreenSourceMIS requires a source proposal sampler and PDF");
                }
            }
        }

        const Point2D source_point =
            sample_uniform_volume(sphere, source_rng);
        if (!scene.visible(sphere.centre, source_point)) return 0.0;
        const double source_value = equation.source(source_point);
        if (source_value == 0.0) return 0.0;
        return throughput * sphere_volume(sphere) * source_value
            * equation.green(sphere, sphere.centre, source_point);
    }
}

template<typename Equation>
double neumann_contribution(const BoundaryScene2D &scene,
                            const Equation &equation,
                            const Sphere2D &sphere,
                            const WoStState2D &state,
                            PRNG &boundary_rng) {
    if (!scene.has_neumann()) return 0.0;
    const NeumannBoundarySample2D sample = sample_neumann_boundary(
        scene, sphere.centre, sphere.radius, boundary_rng);
    if (!sample.valid ||
        !scene.visible(sphere.centre, sample.point)) {
        return 0.0;
    }

    const BoundaryCondition condition =
        equation.boundary(sample.point, sample.boundary_id);
    if (condition.type != BoundaryType::Neumann) {
        throw std::logic_error(
            "boundary scene and equation disagree on a Neumann boundary type");
    }

    const double green = equation.green(
        sphere, sphere.centre, sample.point);
    const double half_space_factor = state.on_neumann ? 2.0 : 1.0;
    return state.throughput * half_space_factor * condition.value
        * green / sample.pdf;
}

template<typename Equation>
PathResult sample_path(const BoundaryScene2D &scene, Point2D start,
                       const Equation &equation,
                       const WoStSettings &settings, PRNG &walk_rng,
                       PRNG &source_rng, PRNG &boundary_rng) {
    WoStState2D state;
    state.position = start;
    const NearestPointResult<2> initial_neumann =
        scene.closest_neumann(start);
    if (initial_neumann.distance <= settings.walk.epsilon) {
        state.on_neumann = true;
        state.neumann_normal =
            scene.outward_normal(initial_neumann.primitive_id);
    }

    while (true) {
        // Keep the same priority in both solvers: invalid numeric state,
        // Dirichlet hit, maximum-step truncation, then another jump.
        if (!std::isfinite(state.value) ||
            !std::isfinite(state.throughput)) {
            return PathResult{
                state.value, state.steps, TerminationReason::Invalid};
        }

        Point2D centre = state.position;
        if (state.on_neumann) {
            centre.x -= settings.walk.epsilon * state.neumann_normal.x;
            centre.y -= settings.walk.epsilon * state.neumann_normal.y;
        }

        const NearestPointResult<2> dirichlet =
            scene.closest_dirichlet(centre);
        if (dirichlet.distance <= settings.walk.epsilon) {
            const BoundaryCondition condition =
                equation.boundary(dirichlet.point, dirichlet.boundary_id);
            state.value += state.throughput * dirichlet_value(condition);
            return PathResult{
                state.value,
                state.steps,
                std::isfinite(state.value)
                    ? TerminationReason::ReachedDirichlet
                    : TerminationReason::Invalid,
            };
        }
        if (state.steps >= settings.walk.max_steps) {
            return PathResult{
                state.value, state.steps,
                TerminationReason::ReachedMaxSteps};
        }

        double geometric_radius = dirichlet.distance;
        const NeumannSilhouetteResult2D silhouette =
            scene.closest_neumann_silhouette(centre);
        if (silhouette.found) {
            geometric_radius = std::min(
                geometric_radius, silhouette.distance);
        }
        double radius = settings.radius_shrink * geometric_radius;
        radius = std::max(radius, settings.minimum_radius);
        if (!(radius > 0.0) || !std::isfinite(radius)) {
            return PathResult{
                state.value, state.steps, TerminationReason::Invalid};
        }

        const Sphere2D sphere{centre, radius};
        state.value += neumann_contribution(
            scene, equation, sphere, state, boundary_rng);
        state.value += source_contribution(
            scene, equation, sphere, state.throughput, source_rng);

        if constexpr (Equation::has_screening) {
            state.throughput *= equation.screening_factor(radius);
        }
        if (!std::isfinite(state.value) ||
            !std::isfinite(state.throughput)) {
            return PathResult{
                state.value, state.steps, TerminationReason::Invalid};
        }

        const Point2D direction = sample_walk_direction(
            walk_rng, state.on_neumann, state.neumann_normal);
        const double hit_epsilon = std::max(
            settings.minimum_radius * 1e-3,
            settings.walk.epsilon * 1e-3);
        const RayHit2D hit = scene.intersect_neumann(
            centre, direction, hit_epsilon, radius);

        state.previous_direction = direction;
        if (hit.hit) {
            state.position = hit.point;
            state.on_neumann = true;
            state.neumann_normal = hit.normal;
        } else {
            state.position = Point2D{
                centre.x + radius * direction.x,
                centre.y + radius * direction.y,
            };
            state.on_neumann = false;
            state.neumann_normal = Point2D{};
        }
        ++state.steps;
    }
}

} // namespace wost_detail

class WoSt {
public:
    explicit WoSt(Settings settings);
    explicit WoSt(WoStSettings settings);

    const WoStSettings &settings() const { return settings_; }

    template<typename Equation>
    Result solve(const BoundaryScene2D &scene, Point2D point,
                 const Equation &equation, PRNG &walk_rng,
                 PRNG &source_rng, PRNG &boundary_rng) const {
        if (!scene.has_dirichlet()) {
            throw std::invalid_argument(
                "WoSt currently requires at least one Dirichlet boundary");
        }

        double mean = 0.0;
        double m2 = 0.0;
        double total_steps = 0.0;
        int successful_paths = 0;
        int max_steps_hits = 0;
        int invalid_paths = 0;

        for (int walk = 0; walk < settings_.walk.walks; ++walk) {
            wost_detail::PathResult path{
                0.0, 0, TerminationReason::Invalid};
            bool accepted = false;
            for (int retry = 0; retry <= settings_.max_retries; ++retry) {
                path = wost_detail::sample_path(
                    scene, point, equation, settings_, walk_rng,
                    source_rng, boundary_rng);
                if (path.termination != TerminationReason::Invalid) {
                    accepted = true;
                    break;
                }
                ++invalid_paths;
            }
            if (!accepted) continue;

            ++successful_paths;
            const double count = static_cast<double>(successful_paths);
            const double delta_before = path.value - mean;
            mean += delta_before / count;
            const double delta_after = path.value - mean;
            m2 += delta_before * delta_after;
            total_steps += path.steps;
            if (path.termination == TerminationReason::ReachedMaxSteps) {
                ++max_steps_hits;
            }
        }

        if (successful_paths == 0) {
            throw std::runtime_error("all WoSt paths were invalid");
        }
        const double variance = successful_paths > 1
            ? m2 / static_cast<double>(successful_paths - 1)
            : 0.0;
        return Result{
            mean,
            variance,
            std::sqrt(variance / static_cast<double>(successful_paths)),
            total_steps / static_cast<double>(successful_paths),
            max_steps_hits,
            invalid_paths,
        };
    }

private:
    WoStSettings settings_;
};

} // namespace wos::solver
