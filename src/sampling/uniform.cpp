#include <cmath>

#include "wos/sampling/uniform.hpp"

namespace wos {

Point2D sample_uniform_surface(const Sphere2D &sphere, PRNG &rng) {
    const double theta = rng.unit() * 2.0 * M_PI;
    return Point2D{
        sphere.centre.x + sphere.radius * std::cos(theta),
        sphere.centre.y + sphere.radius * std::sin(theta),
    };
}

Point3D sample_uniform_surface(const Sphere3D &sphere, PRNG &rng) {
    const double z = 2.0 * rng.unit() - 1.0;
    const double phi = 2.0 * M_PI * rng.unit();
    const double radial = std::sqrt(1.0 - z * z);

    return Point3D{
        sphere.centre.x + sphere.radius * radial * std::cos(phi),
        sphere.centre.y + sphere.radius * radial * std::sin(phi),
        sphere.centre.z + sphere.radius * z,
    };
}

Point2D sample_uniform_volume(const Sphere2D &sphere, PRNG &rng) {
    const double theta = rng.unit() * 2.0 * M_PI;
    const double radius = sphere.radius * std::sqrt(rng.unit_open());

    return Point2D{
        sphere.centre.x + radius * std::cos(theta),
        sphere.centre.y + radius * std::sin(theta),
    };
}

Point3D sample_uniform_volume(const Sphere3D &sphere, PRNG &rng) {
    const double z = 2.0 * rng.unit() - 1.0;
    const double phi = 2.0 * M_PI * rng.unit();
    const double radial = std::sqrt(1.0 - z * z);
    const double radius = sphere.radius * std::cbrt(rng.unit_open());

    return Point3D{
        sphere.centre.x + radius * radial * std::cos(phi),
        sphere.centre.y + radius * radial * std::sin(phi),
        sphere.centre.z + radius * z,
    };
}

} // namespace wos
