// Laplace equation: ∆u = 0 on a domain Ω, with boundary u = g on ∂Ω
#include <cmath>
#include "equations.hpp"
#include "wos/boundary/condition.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/mesh.hpp"
#include "wos/runner.hpp"

namespace wos {

namespace {

struct Laplace2D {
    [[maybe_unused]] static constexpr bool has_source = false;
    [[maybe_unused]] static constexpr bool has_screening = false;

    BoundaryCondition boundary(Point2D p, int boundary_id) const {
        return BoundaryCondition::dirichlet(
            boundary_id == 0 ? p.x*p.x - p.y*p.y : 0.0);
    }

    // Green's function for Laplace operator on 2D spherical domain
    double green(Sphere2D sphere, Point2D x, Point2D y) const {
        return std::log(sphere.radius / dist(x, y)) / (2*M_PI);
    }
};

struct Laplace3D {
    [[maybe_unused]] static constexpr bool has_source = false;
    [[maybe_unused]] static constexpr bool has_screening = false;

    BoundaryCondition boundary(Point3D p) const {
        // combination of spherical harmonics
        double r_sq = p.x*p.x + p.y*p.y;
        double y50  = p.z * (8*p.z*p.z*p.z*p.z - 40*p.z*p.z*r_sq + 15*r_sq*r_sq);
        double y5m3 = (8*p.z*p.z - r_sq) * p.y * (3*p.x*p.x - p.y*p.y);
        return BoundaryCondition::dirichlet(y50 + 4.0 * y5m3);
    }

    // Green's function for Laplace operator on 3D spherical domain
    double green(Sphere3D sphere, Point3D x, Point3D y) const {
        double r = dist(x, y);
        return (sphere.radius - r) / (4*M_PI * r * sphere.radius);
    }
};

int run_2D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    (void)alpha;
    (void)source_mode;
    return run<2>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, Laplace2D{}, seed);
}
int run_3D(int rank, int size, const char *mesh, const char *output, int Nx, int Ny, int Nz,
           int N_walks, double eps, int max_steps, int max_ray_attempts,
           uint64_t seed, double alpha, SourceMode source_mode) {
    (void)alpha;
    (void)source_mode;
    return run<3>(rank, size, mesh, output, Nx, Ny, Nz, N_walks, eps,
                  max_steps, max_ray_attempts, Laplace3D{}, seed);
}

}   // anonymous namespace

// external linkage
const Equation laplace = {
    "laplace",
    run_2D,
    run_3D,
};

}  // namespace wos
