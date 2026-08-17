// Poisson equation: ∆u = -f on a domain Ω, with boundary u = g on ∂Ω
#include <cassert>
#include <cmath>
#include "equations.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"
#include "wos/wos.hpp"

namespace wos {

namespace {

struct Poisson2D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = false;
    [[maybe_unused]] static constexpr bool has_green_source = false;

    static constexpr double load_x = 0.0;
    static constexpr double load_y = 0.0;
    static constexpr double load_radius = 0.15;
    static constexpr double source_strength = 0.1;

    bool source_may_intersect(Sphere2D sphere) const {
        const double dx = sphere.centre.x - load_x;
        const double dy = sphere.centre.y - load_y;
        const double reach = sphere.radius + load_radius;
        return dx*dx + dy*dy <= reach*reach;
    }

    double source(Point2D p) const {
        // Physical demo: a 2 m x 1 m membrane under a uniform circular load.
        // Coordinates are metres and downward displacement is positive.
        // Membrane tension T = 1000 N/m, pressure q = 100 N/m^2, hence
        // the Poisson source f = q/T = 0.1 1/m inside the contact patch.
        const double dx = p.x - load_x;
        const double dy = p.y - load_y;
        return dx*dx + dy*dy <= load_radius*load_radius
             ? source_strength
             : 0.0;
    }

    double boundary(Point2D p, int boundary_id) const {
        // All edges of the membrane are fixed in the reference plane.
        (void)p;
        (void)boundary_id;
        return 0.0;
    }

    // Green's function for Laplace operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        double r = dist(x, y);
        assert(r > 0.0);
        return std::log(sphere.radius / r) / (2*M_PI);
    }
};

struct Poisson3D {
    [[maybe_unused]] static constexpr bool has_source = true;
    [[maybe_unused]] static constexpr bool has_screening = false;
    [[maybe_unused]] static constexpr bool has_green_source = false;

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

    // Green's function for Laplace operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        double r = dist(x, y);
        assert(r > 0.0);
        return (sphere.radius - r) / (4*M_PI * r * sphere.radius);
    }
};

int run_2D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    (void)alpha;
    (void)source_mode;
    return run<2>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, Poisson2D{}, seed);
}
int run_3D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    (void)alpha;
    (void)source_mode;
    return run<3>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, Poisson3D{}, seed);
}

}   // anonymous namespace

// external linkage
const Equation poisson = {
    "poisson",
    run_2D,
    run_3D,
};

}  // namespace wos
