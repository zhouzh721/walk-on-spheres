#pragma once
#include <cmath>

#define EULER_MASCHERONI 0.577215664901532860606512090082402431

namespace wos {

// inlined min, max (no NaN support)
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

// (cylindrical) Bessel function of first kind
// TODO: not very fast. Polynomial approximation better
static inline double bessel_J0(double x) {
    double half_x_sq = 0.25 * x * x;

    double term = 1.0;      // 0th order initialisation
    double sum = 1.0;

    for (int k = 1; k < 200; k++) {
        term *= half_x_sq / (double)(k * k);
        if (term < 1e-8) return sum;    // convergence - exit
        sum += (k % 2 == 0 ? term : -term);
    }

    return sum;
}

// (cylindrical) Bessel function of second kind
// TODO: not very fast
static inline double bessel_Y0(double x) {
    double half_x_sq = 0.25 * x * x;
    double J0_expr = (2.0/M_PI) * (std::log(x/2.0) + EULER_MASCHERONI) * bessel_J0(x);

    double Hk = 0.0;    // kth harmonic number
    double term = 1.0;
    double sum = 0.0;

    for (int k = 1; k < 200; k++) {
        Hk += 1.0/(double)k;
        term *= half_x_sq / (double)(k * k);
        double prod = Hk * term;
        sum += (k % 2 == 0 ? prod : -prod);

        if (prod < 1e-8) break;  // convergence - exit
    }

    return J0_expr - (2.0/M_PI) * sum;
}

// modified Bessel function of first kind
// TODO: not very fast. Polynomial approximation better
static inline double bessel_I0(double x) {
    double half_x_sq = 0.25 * x * x;

    double term = 1.0;      // 0th order initialisation
    double sum = 1.0;

    for (int k = 1; k < 200; k++) {
        term *= half_x_sq / (double)(k * k);
        if (term < 1e-8) return sum;    // convergence - exit
        sum += term;
    }

    return sum;
}

// modified Bessel function of second kind
// TODO: (probably) not very fast
// TODO: only accurate for small values of x - implement asymptotic expression for x >~ 8
// only valid for x > 0
static inline double bessel_K0(double x) {
    double half_x_sq = 0.25 * x * x;
    double I0_expr = -(std::log(x/2.0) + EULER_MASCHERONI) * bessel_I0(x);

    double Hk = 0.0;    // kth harmonic number
    double term = 1.0;
    double sum = 0.0;

    for (int k = 1; k < 200; k++) {
        Hk += 1.0/(double)k;
        term *= half_x_sq / (double)(k * k);
        double prod = Hk * term;
        sum += prod;

        if (prod < 1e-8) break;  // convergence - exit
    }

    return I0_expr + sum;
}

}
