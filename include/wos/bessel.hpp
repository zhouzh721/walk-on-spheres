#pragma once

#include <cmath>
#include <limits>

namespace wos {

namespace bessel_detail {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double asymptotic_start = 50.0;

// Large-x expansions of exp(-x) I_0(x) and exp(x) K_0(x).
// The recurrence generates the standard order-zero coefficients without
// forming either exponentially large or exponentially small values.
inline double asymptotic_sum(double x, bool alternating) {
    double sum = 1.0;
    double term = 1.0;
    double previous = std::numeric_limits<double>::infinity();

    for (int k = 1; k <= 24; ++k) {
        const double odd = static_cast<double>(2 * k - 1);
        double factor = odd * odd / (8.0 * x * static_cast<double>(k));
        if (alternating) factor = -factor;
        term *= factor;

        const double magnitude = std::abs(term);
        if (magnitude > previous) break;
        sum += term;
        if (magnitude <= std::numeric_limits<double>::epsilon() * std::abs(sum)) {
            break;
        }
        previous = magnitude;
    }
    return sum;
}

} // namespace bessel_detail

inline double i0_scaled(double x) {
    x = std::abs(x);
    if (x < bessel_detail::asymptotic_start) {
        return std::exp(-x) * std::cyl_bessel_i(0.0, x);
    }
    return bessel_detail::asymptotic_sum(x, false)
         / std::sqrt(2.0 * bessel_detail::pi * x);
}

inline double k0_scaled(double x) {
    if (x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return std::numeric_limits<double>::infinity();
    if (x < bessel_detail::asymptotic_start) {
        return std::exp(x) * std::cyl_bessel_k(0.0, x);
    }
    return std::sqrt(bessel_detail::pi / (2.0 * x))
         * bessel_detail::asymptotic_sum(x, true);
}

inline double log_i0(double x) {
    x = std::abs(x);
    return x + std::log(i0_scaled(x));
}

inline double log_k0(double x) {
    if (x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return std::numeric_limits<double>::infinity();
    return -x + std::log(k0_scaled(x));
}

inline double i0(double x) {
    const double value = log_i0(x);
    if (value > std::log(std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(value);
}

inline double k0(double x) {
    const double value = log_k0(x);
    if (value == std::numeric_limits<double>::infinity()) {
        return value;
    }
    if (value < std::log(std::numeric_limits<double>::min())) {
        return 0.0;
    }
    return std::exp(value);
}

} // namespace wos
