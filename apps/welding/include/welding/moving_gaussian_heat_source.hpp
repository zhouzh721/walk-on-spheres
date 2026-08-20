#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "wos/mesh.hpp"

namespace welding {

// Time-averaged, through-thickness-uniform Gaussian source for one time step.
// Two-point Gauss-Legendre quadrature follows the continuously moving source
// during the active portion of the step.  Each quadrature component is
// normalized on the finite plate, so the step-integrated absorbed power is
// exactly electrical_power * efficiency * active_fraction.
class MovingGaussianStepSource2D {
public:
    MovingGaussianStepSource2D(
        double xmin, double xmax, double ymin, double ymax,
        double thickness, double electrical_power, double efficiency,
        double sigma_x, double sigma_y, double weld_y,
        double start_x, double speed, double heat_off_time,
        double time_begin, double time_end)
        : absorbed_power_(electrical_power * efficiency) {
        if (!(xmax > xmin) || !(ymax > ymin) || !(thickness > 0.0) ||
            !(electrical_power > 0.0) || !(efficiency > 0.0) ||
            efficiency > 1.0 || !(sigma_x > 0.0) || !(sigma_y > 0.0) ||
            !(speed > 0.0) || !(heat_off_time > 0.0) ||
            !(time_end > time_begin) || weld_y < ymin || weld_y > ymax) {
            throw std::invalid_argument("invalid moving Gaussian parameters");
        }

        const double active_begin = std::max(0.0, time_begin);
        const double active_end = std::min(time_end, heat_off_time);
        if (!(active_end > active_begin)) {
            active_fraction_ = 0.0;
            centre_x_ = start_x + speed * heat_off_time;
            return;
        }

        active_fraction_ =
            (active_end - active_begin) / (time_end - time_begin);
        const double midpoint = 0.5 * (active_begin + active_end);
        const double half_width = 0.5 * (active_end - active_begin);
        constexpr double inverse_sqrt_three =
            0.577350269189625764509148780501957456;
        const std::array<double, 2> times{
            midpoint - inverse_sqrt_three * half_width,
            midpoint + inverse_sqrt_three * half_width,
        };

        centre_x_ = start_x + speed * midpoint;
        for (std::size_t index = 0; index < components_.size(); ++index) {
            Component &component = components_[index];
            component.centre_x = start_x + speed * times[index];
            component.centre_y = weld_y;
            if (component.centre_x < xmin || component.centre_x > xmax) {
                throw std::invalid_argument(
                    "moving Gaussian centre leaves the finite plate");
            }
            const double integral_x = gaussian_integral(
                xmin, xmax, component.centre_x, sigma_x);
            const double integral_y = gaussian_integral(
                ymin, ymax, component.centre_y, sigma_y);
            component.sigma_x = sigma_x;
            component.sigma_y = sigma_y;
            component.coefficient = active_fraction_ * absorbed_power_ /
                (2.0 * thickness * integral_x * integral_y);
        }
    }

    double volumetric_power_density(wos::Point2D point) const {
        if (active_fraction_ == 0.0) {
            return 0.0;
        }
        double value = 0.0;
        for (const Component &component : components_) {
            const double dx =
                (point.x - component.centre_x) / component.sigma_x;
            const double dy =
                (point.y - component.centre_y) / component.sigma_y;
            value += component.coefficient *
                std::exp(-0.5 * (dx * dx + dy * dy));
        }
        return value;
    }

    double active_fraction() const { return active_fraction_; }
    double step_average_absorbed_power() const {
        return active_fraction_ * absorbed_power_;
    }
    double full_absorbed_power() const { return absorbed_power_; }
    double centre_x() const { return centre_x_; }

private:
    struct Component {
        double centre_x = 0.0;
        double centre_y = 0.0;
        double sigma_x = 1.0;
        double sigma_y = 1.0;
        double coefficient = 0.0;
    };

    static double gaussian_integral(double lower, double upper,
                                    double centre, double sigma) {
        constexpr double sqrt_two =
            1.41421356237309504880168872420969808;
        constexpr double sqrt_pi_over_two =
            1.25331413731550025120788264240552263;
        return sigma * sqrt_pi_over_two *
            (std::erf((upper - centre) / (sqrt_two * sigma)) -
             std::erf((lower - centre) / (sqrt_two * sigma)));
    }

    std::array<Component, 2> components_{};
    double absorbed_power_ = 0.0;
    double active_fraction_ = 0.0;
    double centre_x_ = 0.0;
};

} // namespace welding
