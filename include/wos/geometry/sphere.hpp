#pragma once

#include <cmath>

#include "wos/mesh.hpp"

namespace wos {

struct Sphere2D {
    Point2D centre;
    double radius;
};

struct Sphere3D {
    Point3D centre;
    double radius;
};

template<int N>
struct SphereSelector;

template<>
struct SphereSelector<2> {
    using type = Sphere2D;
};

template<>
struct SphereSelector<3> {
    using type = Sphere3D;
};

template<int N>
using Sphere = typename SphereSelector<N>::type;

inline double sphere_volume(const Sphere2D &sphere) {
    return M_PI * sphere.radius * sphere.radius;
}

inline double sphere_volume(const Sphere3D &sphere) {
    const double radius_cubed =
        sphere.radius * sphere.radius * sphere.radius;
    return 4.0 * M_PI * radius_cubed / 3.0;
}

} // namespace wos
