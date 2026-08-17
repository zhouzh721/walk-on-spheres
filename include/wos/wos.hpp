#pragma once
#include <cmath>
#include "wos/bvh.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"
#include "wos/source_mode.hpp"
#include "wos/wos_result.hpp"

namespace wos {

struct Sphere2D { Point2D centre; double radius; };
struct Sphere3D { Point3D centre; double radius; };

// helper to select appropriately dimensioned sphere for Sphere
template<int N> struct SphereSelector;
template<> struct SphereSelector<2> { using type = Sphere2D; };
template<> struct SphereSelector<3> { using type = Sphere3D; };
template<int N> using Sphere = typename SphereSelector<N>::type;

inline double sphere_volume(Sphere2D s) {
    return M_PI * s.radius * s.radius;
}
inline double sphere_volume(Sphere3D s) {
    double rcubed = s.radius * s.radius * s.radius;
    return 4.0 * M_PI * rcubed / 3.0;
}

Point2D step(Sphere2D sphere, PRNG &rng);
Point3D step(Sphere3D sphere, PRNG &rng);

Point2D sphere_sample(Sphere2D sphere, PRNG &rng);
Point3D sphere_sample(Sphere3D sphere, PRNG &rng);

template<int N>
struct StartPoint {
    double radius;
    Point<N> nearest;
    int boundary_id;
};

template<int N>
StartPoint<N> get_start_point(const BVH<N> &bvh, Point<N> p0) {
    if constexpr (N == 2) {
        auto nearest = bvh_npq(bvh, p0);
        return StartPoint<N>{nearest.distance, nearest.point, nearest.boundary_id};
    } else {
        Point<N> nearest;
        double radius = bvh_npq(bvh, p0, &nearest);
        return StartPoint<N>{radius, nearest, -1};
    }
}

template<int N, typename Eq>
SampleResult wos_sample(const BVH<N> &bvh, Point<N> p0, const StartPoint<N> &start,
                        const Eq &eq, double epsilon, PRNG &walk_rng,
                        PRNG &source_rng,
                        int max_steps = 1'000) {
    Point<N> p = p0;
    Point<N> np = start.nearest;
    [[maybe_unused]] int boundary_id = start.boundary_id;
    Sphere<N> sphere{p, start.radius};

    double sample = 0.0;
    double weight = 1.0;
    int steps = 0;

    while (sphere.radius > epsilon && steps < max_steps) {
        if constexpr (Eq::has_source) {
            bool source_may_intersect = true;
            if constexpr (N == 2) {
                source_may_intersect = eq.source_may_intersect(sphere);
            }
            if (source_may_intersect) {
                if constexpr (Eq::has_green_source) {
                    if (eq.source_mode == SourceMode::Green) {
                        Point<N> sp = eq.sample_green(sphere, source_rng);
                        const double source_value = eq.source(sp);
                        if (source_value != 0.0) {
                            sample += weight * eq.green_mass(sphere.radius)
                                    * source_value;
                        }
                    } else {
                        Point<N> sp = sphere_sample(sphere, source_rng);
                        const double source_value = eq.source(sp);
                        if (source_value != 0.0) {
                            sample += weight * sphere_volume(sphere) * source_value
                                    * eq.green(sphere, p, sp);
                        }
                    }
                } else {
                    Point<N> sp = sphere_sample(sphere, source_rng);
                    const double source_value = eq.source(sp);
                    if (source_value != 0.0) {
                        sample += weight * sphere_volume(sphere) * source_value
                                * eq.green(sphere, p, sp);
                    }
                }
            }
        }
        if constexpr (Eq::has_screening) {
            weight *= eq.screening_factor(sphere.radius);
        }

        p = step(sphere, walk_rng);
        ++steps;
        double radius;
        if constexpr (N == 2) {
            auto nearest = bvh_npq(bvh, p);
            radius = nearest.distance;
            np = nearest.point;
            boundary_id = nearest.boundary_id;
        } else {
            radius = bvh_npq(bvh, p, &np);
        }
        sphere = Sphere<N>{p, radius};
    }

    const bool max_steps_reached = sphere.radius > epsilon;
    // If the safety limit is reached, finish the approximation by projecting
    // the current walk position to its nearest boundary point.
    if constexpr (N == 2) {
        sample += weight * eq.boundary(np, boundary_id);
    } else {
        sample += weight * eq.boundary(np);
    }
    return SampleResult{sample, steps, max_steps_reached};
}

// One Monte Carlo walk-on-spheres sample starting at p0.
template<int N, typename Eq>
SampleResult wos_sample_once(const BVH<N> &bvh, Point<N> p0, const Eq &eq,
                             double epsilon, PRNG &walk_rng, PRNG &source_rng,
                             int max_steps = 1'000) {
    auto start = get_start_point(bvh, p0);
    return wos_sample(bvh, p0, start, eq, epsilon,
                      walk_rng, source_rng, max_steps);
}

// Backwards-compatible overload that keeps source sampling and WoS steps on
// the same random stream.
template<int N, typename Eq>
SampleResult wos_sample_once(const BVH<N> &bvh, Point<N> p0, const Eq &eq,
                             double epsilon, PRNG &rng, int max_steps = 1'000) {
    return wos_sample_once(bvh, p0, eq, epsilon, rng, rng, max_steps);
}

// Monte Carlo walk-on-spheres estimator for u(p0).
// Eq should specify source, screening, boundary, etc. (as appropriate).
template<int N, typename Eq>
WoSResult wos_run(const BVH<N> &bvh, Point<N> p0, const StartPoint<N> &start,
                  const Eq &eq, int N_walks, double epsilon,
                  PRNG &walk_rng, PRNG &source_rng, int max_steps = 1'000) {
    double mean = 0.0;
    double m2 = 0.0;
    double total_steps = 0.0;
    int max_steps_hits = 0;

    for (int i = 0; i < N_walks; i++) {
        SampleResult sample =
            wos_sample(bvh, p0, start, eq, epsilon,
                       walk_rng, source_rng, max_steps);
        double count = static_cast<double>(i + 1);
        double delta_before = sample.value - mean;
        mean += delta_before / count;
        double delta_after = sample.value - mean;
        m2 += delta_before * delta_after;
        total_steps += sample.steps;
        if (sample.max_steps_reached) ++max_steps_hits;
    }

    double variance = N_walks > 1 ? m2 / static_cast<double>(N_walks - 1) : 0.0;
    double standard_error = N_walks > 0
                          ? std::sqrt(variance / static_cast<double>(N_walks))
                          : 0.0;
    double mean_steps = N_walks > 0
                      ? total_steps / static_cast<double>(N_walks)
                      : 0.0;

    return WoSResult{mean, variance, standard_error, mean_steps, max_steps_hits};
}

// Backwards-compatible overload that keeps source sampling and WoS steps on
// the same random stream.
template<int N, typename Eq>
WoSResult wos_run(const BVH<N> &bvh, Point<N> p0, const StartPoint<N> &start,
                  const Eq &eq, int N_walks, double epsilon,
                  PRNG &rng, int max_steps = 1'000) {
    return wos_run(bvh, p0, start, eq, N_walks, epsilon, rng, rng, max_steps);
}

}
