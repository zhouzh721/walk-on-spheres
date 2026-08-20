#pragma once

#include "wos/geometry/sphere.hpp"
#include "wos/prng.hpp"

namespace wos {

Point2D sample_uniform_surface(const Sphere2D &sphere, PRNG &rng);
Point3D sample_uniform_surface(const Sphere3D &sphere, PRNG &rng);

Point2D sample_uniform_volume(const Sphere2D &sphere, PRNG &rng);
Point3D sample_uniform_volume(const Sphere3D &sphere, PRNG &rng);

} // namespace wos
