#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <ostream>
#include <vector>

#include "welding/grid_utils.hpp"
#include "wos/grid.hpp"

namespace welding {

struct TemperatureFieldSummary2D {
    double peak_rise = 0.0;
    int peak_i = 0;
    int peak_j = 0;
    double mean_standard_error = 0.0;
    double max_standard_error = 0.0;
    double minimum_interior_rise =
        std::numeric_limits<double>::infinity();
    std::size_t negative_interior_points = 0;
    double thermal_energy_rise = 0.0;
};

inline TemperatureFieldSummary2D summarize_temperature_field_2d(
        const wos::Grid &grid, const std::vector<double> &field,
        const std::vector<double> &standard_error, double density,
        double specific_heat, double thickness) {
    TemperatureFieldSummary2D summary;
    double standard_error_sum = 0.0;
    double temperature_integral = 0.0;
    std::size_t interior_count = 0;
    const double dx = (grid.xmax - grid.xmin) /
        static_cast<double>(grid.Nx - 1);
    const double dy = (grid.ymax - grid.ymin) /
        static_cast<double>(grid.Ny - 1);

    for (int i = 0; i < grid.Nx; ++i) {
        const double wx = (i == 0 || i == grid.Nx - 1) ? 0.5 : 1.0;
        for (int j = 0; j < grid.Ny; ++j) {
            const double wy = (j == 0 || j == grid.Ny - 1) ? 0.5 : 1.0;
            const std::size_t index = flat_index_2d(i, j, grid.Ny);
            const double value = field[index];
            temperature_integral += wx * wy * value;
            if (value > summary.peak_rise) {
                summary.peak_rise = value;
                summary.peak_i = i;
                summary.peak_j = j;
            }
            if (!is_boundary_index_2d(grid, i, j)) {
                standard_error_sum += standard_error[index];
                summary.max_standard_error = std::max(
                    summary.max_standard_error, standard_error[index]);
                summary.minimum_interior_rise = std::min(
                    summary.minimum_interior_rise, value);
                if (value < 0.0) ++summary.negative_interior_points;
                ++interior_count;
            }
        }
    }

    if (interior_count > 0) {
        summary.mean_standard_error = standard_error_sum /
            static_cast<double>(interior_count);
    } else {
        summary.minimum_interior_rise = 0.0;
    }
    summary.thermal_energy_rise = density * specific_heat * thickness *
        temperature_integral * dx * dy;
    return summary;
}

inline void write_temperature_field_2d(
        std::ostream &output, int step, double time,
        double ambient_temperature, const wos::Grid &grid,
        const std::vector<double> &field,
        const std::vector<double> &standard_error) {
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = flat_index_2d(i, j, grid.Ny);
            output << step << ',' << time << ',' << x << ',' << y << ','
                   << field[index] << ','
                   << ambient_temperature + field[index] << ','
                   << standard_error[index] << '\n';
        }
    }
}

} // namespace welding
