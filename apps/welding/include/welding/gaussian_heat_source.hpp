#pragma once

#include <cmath>
#include <stdexcept>

#include "wos/mesh.hpp"

namespace welding {

// Two-dimensional Gaussian representation of a through-thickness-uniform
// volumetric heat source. The finite rectangular-domain normalization makes
// thickness * integral(q''' dA) exactly equal to efficiency * electrical_power.
class GaussianHeatSource2D {
public:
    GaussianHeatSource2D(double xmin, double xmax,
                         double ymin, double ymax,
                         double thickness,
                         double electrical_power,
                         double efficiency,
                         double centre_x, double centre_y,
                         double sigma_x, double sigma_y)
        : xmin_(xmin), xmax_(xmax), ymin_(ymin), ymax_(ymax),
          thickness_(thickness), absorbed_power_(electrical_power * efficiency),
          centre_x_(centre_x), centre_y_(centre_y),
          sigma_x_(sigma_x), sigma_y_(sigma_y),
          integral_x_(gaussian_integral(xmin, xmax, centre_x, sigma_x)),
          integral_y_(gaussian_integral(ymin, ymax, centre_y, sigma_y)) {
        if (!(xmax > xmin) || !(ymax > ymin) || !(thickness > 0.0) ||
            !(electrical_power > 0.0) || !(efficiency > 0.0) ||
            efficiency > 1.0 || !(sigma_x > 0.0) || !(sigma_y > 0.0) ||
            centre_x < xmin || centre_x > xmax ||
            centre_y < ymin || centre_y > ymax) {
            throw std::invalid_argument("invalid Gaussian heat-source parameters");
        }
    }

    double volumetric_power_density(wos::Point2D point) const {
        const double dx = (point.x - centre_x_) / sigma_x_;
        const double dy = (point.y - centre_y_) / sigma_y_;
        const double shape = std::exp(-0.5 * (dx * dx + dy * dy));
        return absorbed_power_ * shape /
               (thickness_ * integral_x_ * integral_y_);
    }

    double integrated_power_midpoint(int nx, int ny) const {
        if (nx <= 0 || ny <= 0) {
            throw std::invalid_argument(
                "midpoint integration requires positive sample counts");
        }
        const double dx = (xmax_ - xmin_) / static_cast<double>(nx);
        const double dy = (ymax_ - ymin_) / static_cast<double>(ny);
        double integral = 0.0;
        for (int i = 0; i < nx; ++i) {
            const double x = xmin_ + (static_cast<double>(i) + 0.5) * dx;
            for (int j = 0; j < ny; ++j) {
                const double y = ymin_ + (static_cast<double>(j) + 0.5) * dy;
                integral += volumetric_power_density({x, y});
            }
        }
        return thickness_ * integral * dx * dy;
    }

    double absorbed_power() const { return absorbed_power_; }
    double peak_power_density() const {
        return absorbed_power_ / (thickness_ * integral_x_ * integral_y_);
    }
    double integral_x() const { return integral_x_; }
    double integral_y() const { return integral_y_; }
    double centre_x() const { return centre_x_; }
    double centre_y() const { return centre_y_; }
    double sigma_x() const { return sigma_x_; }
    double sigma_y() const { return sigma_y_; }

private:
    static double gaussian_integral(double lower, double upper,
                                    double centre, double sigma) {
        if (!(upper > lower) || !(sigma > 0.0)) {
            throw std::invalid_argument("invalid Gaussian integration interval");
        }
        constexpr double sqrt_two = 1.41421356237309504880168872420969808;
        constexpr double sqrt_pi_over_two =
            1.25331413731550025120788264240552263;
        return sigma * sqrt_pi_over_two *
               (std::erf((upper - centre) / (sqrt_two * sigma)) -
                std::erf((lower - centre) / (sqrt_two * sigma)));
    }

    double xmin_;
    double xmax_;
    double ymin_;
    double ymax_;
    double thickness_;
    double absorbed_power_;
    double centre_x_;
    double centre_y_;
    double sigma_x_;
    double sigma_y_;
    double integral_x_;
    double integral_y_;
};

} // namespace welding
