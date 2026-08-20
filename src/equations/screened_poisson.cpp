// Screened Poisson equation: ∆u - α²u = -f on a domain Ω, with boundary u = g on ∂Ω
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "equations.hpp"
#include "wos/boundary/condition.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"
#include "wos/sampling/screened_green.hpp"
#include "wos/source_mode.hpp"

namespace wos {

namespace {

// The usual nearest-boundary termination drops the final ball's screened
// attenuation and source contribution. Keep that approximation small by
// requiring the terminal ball to also satisfy alpha * radius <= 0.05.
constexpr double max_terminal_alpha_radius = 0.05;

double screened_epsilon(int rank, double requested_epsilon, double alpha) {
    const double screening_limited_epsilon =
        max_terminal_alpha_radius / alpha;
    const double effective_epsilon =
        std::min(requested_epsilon, screening_limited_epsilon);

    if (rank == 0 && effective_epsilon < requested_epsilon) {
        std::printf(
            "Screened stopping distance reduced:\n"
            "  requested epsilon:       %.12g\n"
            "  effective epsilon:       %.12g\n"
            "  alpha * effective eps:   %.12g\n",
            requested_epsilon, effective_epsilon,
            alpha * effective_epsilon);
    }
    return effective_epsilon;
}

struct ScreenedPoisson2D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = true;
    [[maybe_unused]] static constexpr bool has_green_source = true;

    double alpha = 5.0;
    SourceMode source_mode = SourceMode::Uniform;

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

    BoundaryCondition boundary(Point2D p, int boundary_id) const {
        return BoundaryCondition::dirichlet(
            boundary_id == 0 ? p.x*p.x - p.y*p.y : 0.0);
    }

    // Green's function for screened Poisson operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        const double r = dist(x, y);
        assert(r > 0.0);
        return screened_green::green_2d(alpha, sphere.radius, r);
    }

    // weight factor: 1 / I_0(αr) for the screened Poisson eq in 2D
    double screening_factor(double radius) const {
        return screened_green::weight_2d(alpha, radius);
    }

    double green_mass(double radius) const {
        return screened_green::mass_2d(alpha, radius);
    }

    Point2D sample_green(Sphere2D sphere, PRNG &rng) const {
        const double radius = screened_green::sample_radius_2d(
            alpha, sphere.radius, rng);
        const double angle = 2.0 * screened_green::pi * rng.unit();
        return Point2D{
            sphere.centre.x + radius * std::cos(angle),
            sphere.centre.y + radius * std::sin(angle),
        };
    }
};

struct ScreenedPoisson3D {
    [[maybe_unused]] static constexpr bool has_source = false;
    [[maybe_unused]] static constexpr bool has_screening = true;
    [[maybe_unused]] static constexpr bool has_green_source = true;

    double alpha = 5.0;
    SourceMode source_mode = SourceMode::Uniform;

    double source(Point3D p) const {
        (void)p;
        return 0.0;
    }

    BoundaryCondition boundary(Point3D p) const {
        // combination of spherical harmonics
        double r_sq = p.x*p.x + p.y*p.y;
        double y50  = p.z * (8*p.z*p.z*p.z*p.z - 40*p.z*p.z*r_sq + 15*r_sq*r_sq);
        double y5m3 = (8*p.z*p.z - r_sq) * p.y * (3*p.x*p.x - p.y*p.y);
        return BoundaryCondition::dirichlet(y50 + 4.0 * y5m3);
    }

    // Green's function for screened Poisson operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        const double r = dist(x, y);
        assert(r > 0.0);
        return screened_green::green_3d(alpha, sphere.radius, r);
    }

    // weight factor
    double screening_factor(double radius) const {
        return screened_green::weight_3d(alpha, radius);
    }

    double green_mass(double radius) const {
        return screened_green::mass_3d(alpha, radius);
    }

    Point3D sample_green(Sphere3D sphere, PRNG &rng) const {
        const double radius = screened_green::sample_radius_3d(
            alpha, sphere.radius, rng);
        const double z = 2.0 * rng.unit() - 1.0;
        const double angle = 2.0 * screened_green::pi * rng.unit();
        const double xy = std::sqrt(1.0 - z * z);
        return Point3D{
            sphere.centre.x + radius * xy * std::cos(angle),
            sphere.centre.y + radius * xy * std::sin(angle),
            sphere.centre.z + radius * z,
        };
    }
};

int run_2D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    const double effective_eps = screened_epsilon(rank, eps, alpha);
    return run<2>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, effective_eps,
                  max_steps, max_ray_attempts,
                  ScreenedPoisson2D{alpha, source_mode}, seed,
                  true, alpha, source_mode);
}
int run_3D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    const double effective_eps = screened_epsilon(rank, eps, alpha);
    return run<3>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, effective_eps,
                  max_steps, max_ray_attempts,
                  ScreenedPoisson3D{alpha, source_mode}, seed,
                  true, alpha, source_mode);
}

}   // anonymous namespace

// external linkage
const Equation screened_poisson = {
    "screened_poisson",
    run_2D,
    run_3D,
};

}  // namespace wos
