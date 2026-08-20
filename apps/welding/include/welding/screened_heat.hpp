#pragma once

#include <cmath>

#include "wos/boundary/condition.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/prng.hpp"
#include "wos/sampling/screened_green.hpp"
#include "wos/source_mode.hpp"

namespace welding {

// Common screened-Poisson operator produced by implicit-Euler heat steps with
// homogeneous Dirichlet temperature-rise boundaries. Derived equations only
// provide source(point).
class ZeroDirichletScreenedHeat2D {
public:
    static constexpr bool has_source = true;
    static constexpr bool has_screening = true;
    static constexpr bool has_green_source = true;

    explicit ZeroDirichletScreenedHeat2D(double alpha_value)
        : alpha_(alpha_value) {}

    bool source_may_intersect([[maybe_unused]] wos::Sphere2D sphere) const {
        return true;
    }

    wos::BoundaryType boundary_type(
            [[maybe_unused]] int boundary_id) const {
        return wos::BoundaryType::Dirichlet;
    }

    wos::BoundaryCondition boundary(
            [[maybe_unused]] wos::Point2D point,
            [[maybe_unused]] int boundary_id) const {
        return wos::BoundaryCondition::dirichlet(0.0);
    }

    double green(wos::Sphere2D sphere, wos::Point2D x,
                 wos::Point2D y) const {
        return wos::screened_green::green_2d(
            alpha_, sphere.radius, wos::dist(x, y));
    }

    double screening_factor(double radius) const {
        return wos::screened_green::weight_2d(alpha_, radius);
    }

    double green_mass(double radius) const {
        return wos::screened_green::mass_2d(alpha_, radius);
    }

    wos::Point2D sample_green(wos::Sphere2D sphere, wos::PRNG &rng) const {
        const double radius = wos::screened_green::sample_radius_2d(
            alpha_, sphere.radius, rng);
        const double angle = 2.0 * wos::screened_green::pi * rng.unit();
        return wos::Point2D{
            sphere.centre.x + radius * std::cos(angle),
            sphere.centre.y + radius * std::sin(angle),
        };
    }

    double screening_parameter() const { return alpha_; }
    double screening_parameter_squared() const { return alpha_ * alpha_; }

    wos::SourceMode source_mode = wos::SourceMode::Green;

private:
    double alpha_;
};

} // namespace welding
