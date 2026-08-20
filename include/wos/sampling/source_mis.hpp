#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "wos/mesh.hpp"
#include "wos/prng.hpp"

namespace wos::source_mis {

// A source proposal is an optional equation capability. Detect the actual
// interface instead of requiring a second boolean alongside
// Equation::has_source.
template<typename Equation, typename PointType, typename = void>
struct HasSourceProposal : std::false_type {};

template<typename Equation, typename PointType>
struct HasSourceProposal<Equation, PointType, std::void_t<
    decltype(std::declval<const Equation &>().sample_source_proposal(
        std::declval<PRNG &>())),
    decltype(std::declval<const Equation &>().source_proposal_pdf(
        std::declval<PointType>())),
    decltype(std::declval<const Equation &>().source_mis_green_probability())
>> : std::true_type {};

template<typename Equation, typename PointType>
inline constexpr bool has_source_proposal_v =
    HasSourceProposal<Equation, PointType>::value;

// Estimate the Green source integral on one WoS sphere using a random-mixture
// balance heuristic. The source proposal is normalized on the full physical
// domain; samples outside the current sphere contribute zero through the
// integral's sphere indicator.
template<typename Equation, typename SphereType>
double green_source_mis_contribution(const Equation &equation,
                                     const SphereType &sphere,
                                     PRNG &rng) {
    using PointType = std::decay_t<decltype(sphere.centre)>;
    static_assert(has_source_proposal_v<Equation, PointType>,
                  "Green-source MIS requires a source proposal sampler and PDF");

    const double beta = equation.source_mis_green_probability();
    if (!(beta > 0.0 && beta < 1.0) || !std::isfinite(beta)) {
        throw std::invalid_argument(
            "Green-source MIS probability must be finite and in (0, 1)");
    }

    const PointType sample_point = rng.unit() < beta
        ? equation.sample_green(sphere, rng)
        : equation.sample_source_proposal(rng);
    const double source_distance = dist(sphere.centre, sample_point);
    if (!(source_distance > 0.0 && source_distance < sphere.radius)) {
        return 0.0;
    }

    const double green_mass = equation.green_mass(sphere.radius);
    const double green_value = equation.green(
        sphere, sphere.centre, sample_point);
    if (!(green_mass > 0.0) || !std::isfinite(green_mass) ||
        !(green_value >= 0.0) || !std::isfinite(green_value)) {
        throw std::runtime_error("invalid Green density for source MIS");
    }

    const double green_pdf = green_value / green_mass;
    const double source_pdf = equation.source_proposal_pdf(sample_point);
    const double mixture_pdf =
        beta * green_pdf + (1.0 - beta) * source_pdf;
    if (!(source_pdf >= 0.0) || !std::isfinite(source_pdf) ||
        !(mixture_pdf > 0.0) || !std::isfinite(mixture_pdf)) {
        throw std::runtime_error("invalid Green-source MIS density");
    }

    const double source_value = equation.source(sample_point);
    return source_value == 0.0
        ? 0.0
        : green_value * source_value / mixture_pdf;
}

} // namespace wos::source_mis
