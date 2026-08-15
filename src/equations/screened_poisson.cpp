// Screened Poisson equation: ∆u - α²u = -f on a domain Ω, with boundary u = g on ∂Ω
#include <cassert>
#include <cmath>
#include "equations.hpp"
#include "wos/fastmath.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"
#include "wos/wos.hpp"

using namespace wos;

namespace {

struct ScreenedPoisson2D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = true;
    static constexpr double alpha = 5.0;

    bool source_may_intersect(Sphere2D sphere) const {
        constexpr double source_x = 0.45;
        constexpr double source_y = 0.0;
        constexpr double source_radius = 0.5;
        const double dx = sphere.centre.x - source_x;
        const double dy = sphere.centre.y - source_y;
        const double reach = sphere.radius + source_radius;
        return dx*dx + dy*dy <= reach*reach;
    }

    double source(Point2D p) const {
        // a bump centred at (0.45, 0) with radius 0.5
        return std::sqrt((p.x-0.45)*(p.x-0.45) + p.y*p.y) <= 0.5 ? 5.0 : 0.0;
    }

    double boundary(Point2D p, int boundary_id) const {
        return boundary_id == 0 ? p.x*p.x - p.y*p.y : 0.0;
    }

    // Green's function for screened Poisson operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        double r = dist(x, y);
        assert(r > 0.0);
        double prod = r * alpha;
        double prod_R = sphere.radius * alpha;
        return (bessel_K0(prod) - bessel_I0(prod) * bessel_K0(prod_R) / bessel_I0(prod_R)) / (2*M_PI);
    }

    // weight factor: 1 / I_0(αr) for the screened Poisson eq in 2D
    double screening_factor(double radius) const {
        double prod = radius * alpha;
        if (prod < 1e-8) return 1.0;
        return 1.0 / bessel_I0(prod);
    }
};

struct ScreenedPoisson3D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = true;
    static constexpr double alpha = 5.0;

    double source(Point3D p) const {
        (void)p;
        return 0.0;
    }

    double boundary(Point3D p) const {
        // combination of spherical harmonics
        double r_sq = p.x*p.x + p.y*p.y;
        double y50  = p.z * (8*p.z*p.z*p.z*p.z - 40*p.z*p.z*r_sq + 15*r_sq*r_sq);
        double y5m3 = (8*p.z*p.z - r_sq) * p.y * (3*p.x*p.x - p.y*p.y);
        return y50 + 4.0 * y5m3;
    }

    // Green's function for screened Poisson operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        double r = dist(x, y);
        assert(r > 0.0);
        double prod = r * alpha;
        double prod_R = sphere.radius * alpha;
        return (std::sinh(prod_R - prod)) / (4*M_PI * r * std::sinh(prod_R));
    }

    // weight factor
    double screening_factor(double radius) const {
        double prod = radius * alpha;
        if (prod < 1e-8) return 1.0;
        return prod / std::sinh(prod);
    }
};

int run_2D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed) {
    return run<2>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, ScreenedPoisson2D{}, seed);
}
int run_3D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed) {
    return run<3>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, ScreenedPoisson3D{}, seed);
}

}   // anonymous namespace

// external linkage
const Equation screened_poisson = {
    "screened_poisson",
    run_2D,
    run_3D,
};
