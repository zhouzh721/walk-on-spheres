#pragma once

#include <cmath>
#include <stdexcept>

#include "wos/boundary/condition.hpp"
#include "wos/geometry/scene.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/prng.hpp"
#include "wos/sampling/source_mis.hpp"
#include "wos/sampling/uniform.hpp"
#include "wos/solver/result.hpp"
#include "wos/solver/settings.hpp"
#include "wos/solver/start_point.hpp"
#include "wos/source_mode.hpp"

namespace wos::solver {

namespace detail {

template<int N, typename Equation>
SampleResult sample(const GeometryScene<N> &scene, Point<N> start_position,
                    const StartPoint<N> &start, const Equation &equation,
                    double epsilon, PRNG &walk_rng, PRNG &source_rng,
                    int max_steps) {
    Point<N> position = start_position;
    Point<N> nearest = start.nearest;
    [[maybe_unused]] int boundary_id = start.boundary_id;
    Sphere<N> sphere{position, start.radius};

    double sample_value = 0.0;
    double weight = 1.0;
    int steps = 0;

    while (true) {
        // A boundary hit takes precedence when the final allowed jump lands
        // inside the epsilon shell.
        if (sphere.radius <= epsilon) {
            if constexpr (N == 2) {
                sample_value += weight * dirichlet_value(
                    equation.boundary(nearest, boundary_id));
            } else {
                sample_value += weight * dirichlet_value(
                    equation.boundary(nearest));
            }
            return SampleResult{
                sample_value, steps, TerminationReason::ReachedDirichlet};
        }
        if (steps >= max_steps) {
            // Preserve the existing WoS truncation policy: approximate the
            // unresolved tail with the nearest boundary value.
            if constexpr (N == 2) {
                sample_value += weight * dirichlet_value(
                    equation.boundary(nearest, boundary_id));
            } else {
                sample_value += weight * dirichlet_value(
                    equation.boundary(nearest));
            }
            return SampleResult{
                sample_value, steps, TerminationReason::ReachedMaxSteps};
        }

        if constexpr (Equation::has_source) {
            bool source_may_intersect = true;
            if constexpr (N == 2) {
                source_may_intersect = equation.source_may_intersect(sphere);
            }
            if (source_may_intersect) {
                if constexpr (Equation::has_green_source) {
                    if (equation.source_mode == SourceMode::Green) {
                        const Point<N> source_point =
                            equation.sample_green(sphere, source_rng);
                        const double source_value =
                            equation.source(source_point);
                        if (source_value != 0.0) {
                            sample_value += weight
                                * equation.green_mass(sphere.radius)
                                * source_value;
                        }
                    } else if (equation.source_mode ==
                               SourceMode::GreenSourceMIS) {
                        if constexpr (source_mis::has_source_proposal_v<
                                          Equation, Point<N>>) {
                            sample_value += weight
                                * source_mis::green_source_mis_contribution(
                                    equation, sphere, source_rng);
                        } else {
                            throw std::invalid_argument(
                                "GreenSourceMIS requires a source proposal sampler and PDF");
                        }
                    } else {
                        const Point<N> source_point =
                            sample_uniform_volume(sphere, source_rng);
                        const double source_value =
                            equation.source(source_point);
                        if (source_value != 0.0) {
                            sample_value += weight * sphere_volume(sphere)
                                * source_value
                                * equation.green(
                                    sphere, position, source_point);
                        }
                    }
                } else {
                    const Point<N> source_point =
                        sample_uniform_volume(sphere, source_rng);
                    const double source_value = equation.source(source_point);
                    if (source_value != 0.0) {
                        sample_value += weight * sphere_volume(sphere)
                            * source_value
                            * equation.green(sphere, position, source_point);
                    }
                }
            }
        }

        if constexpr (Equation::has_screening) {
            weight *= equation.screening_factor(sphere.radius);
        }

        position = sample_uniform_surface(sphere, walk_rng);
        ++steps;

        const NearestPointResult<N> boundary =
            scene.closest_boundary(position);
        nearest = boundary.point;
        boundary_id = boundary.boundary_id;
        sphere = Sphere<N>{position, boundary.distance};
    }
}

} // namespace detail

class WoS {
public:
    explicit WoS(Settings settings);

    const Settings &settings() const {
        return settings_;
    }

    template<int N, typename Equation>
    Result solve(const GeometryScene<N> &scene, Point<N> point,
                 const StartPoint<N> &start, const Equation &equation,
                 PRNG &walk_rng, PRNG &source_rng) const {
        double mean = 0.0;
        double m2 = 0.0;
        double total_steps = 0.0;
        int max_steps_hits = 0;

        for (int i = 0; i < settings_.walks; ++i) {
            const SampleResult path = detail::sample(
                scene, point, start, equation, settings_.epsilon,
                walk_rng, source_rng, settings_.max_steps);
            const double count = static_cast<double>(i + 1);
            const double delta_before = path.value - mean;
            mean += delta_before / count;
            const double delta_after = path.value - mean;
            m2 += delta_before * delta_after;
            total_steps += path.steps;
            if (path.termination == TerminationReason::ReachedMaxSteps) {
                ++max_steps_hits;
            }
        }

        const double variance = settings_.walks > 1
            ? m2 / static_cast<double>(settings_.walks - 1)
            : 0.0;
        const double standard_error = std::sqrt(
            variance / static_cast<double>(settings_.walks));
        const double mean_steps =
            total_steps / static_cast<double>(settings_.walks);

        return Result{
            mean, variance, standard_error, mean_steps, max_steps_hits};
    }

    template<int N, typename Equation>
    Result solve(const GeometryScene<N> &scene, Point<N> point,
                 const Equation &equation, PRNG &walk_rng,
                 PRNG &source_rng) const {
        return solve(scene, point, find_start_point(scene, point), equation,
                     walk_rng, source_rng);
    }

private:
    Settings settings_;
};

} // namespace wos::solver
