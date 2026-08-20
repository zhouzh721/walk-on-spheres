#pragma once

#include <cmath>
#include <limits>

#include "wos/prng.hpp"

namespace wos::poisson_green {

constexpr double pi = 3.141592653589793238462643383279502884;

inline double green_2d(double sphere_radius, double distance) {
    if (!(distance > 0.0) || distance > sphere_radius || sphere_radius <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (distance == sphere_radius) return 0.0;
    return std::log(sphere_radius / distance) / (2.0 * pi);
}

inline double mass_2d(double sphere_radius) {
    return sphere_radius * sphere_radius / 4.0;
}

// The normalized 2D radial density is 4*s*log(1/s), with s=r/R.
// A product of two independent uniforms has density -log(u), hence
// r/R = sqrt(U*V) samples the Green-weighted radius directly.
inline double sample_radius_2d(double sphere_radius, PRNG &rng) {
    return sphere_radius *
        std::sqrt(rng.unit_open() * rng.unit_open());
}

inline double green_3d(double sphere_radius, double distance) {
    if (!(distance > 0.0) || distance > sphere_radius || sphere_radius <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (distance == sphere_radius) return 0.0;
    return (sphere_radius - distance) /
           (4.0 * pi * distance * sphere_radius);
}

inline double mass_3d(double sphere_radius) {
    return sphere_radius * sphere_radius / 6.0;
}

// The normalized 3D radius r/R follows Beta(2, 2). Generate its two Gamma(2)
// components from products of independent uniforms.
inline double sample_radius_3d(double sphere_radius, PRNG &rng) {
    const double a = -std::log(rng.unit_open() * rng.unit_open());
    const double b = -std::log(rng.unit_open() * rng.unit_open());
    return sphere_radius * a / (a + b);
}

} // namespace wos::poisson_green
