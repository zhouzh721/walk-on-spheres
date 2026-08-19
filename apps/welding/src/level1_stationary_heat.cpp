#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <mpi.h>

#include "welding/gaussian_heat_source.hpp"
#include "welding/rectangular_plate.hpp"
#include "wos/bvh.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/mesh.hpp"
#include "wos/poisson.hpp"
#include "wos/prng.hpp"
#include "wos/source_mode.hpp"
#include "wos/wos.hpp"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

#ifndef WELDING_RESULTS_DIR
#define WELDING_RESULTS_DIR "apps/welding/results"
#endif

// Level 1 intentionally uses one documented built-in case. Runtime material
// and process configuration is deferred until the physics and units pass the
// validation gates.
struct Level1Config {
    int nx = 49;
    int ny = 33;
    int walks = 6'000;
    int max_steps = 1'000;
    double epsilon = 8e-6;
    std::uint64_t seed = 20260818ULL;

    double length_x = 0.12;
    double length_y = 0.08;
    double thickness = 0.006;
    double conductivity = 35.0;
    double ambient_temperature = 293.15;

    double electrical_power = 800.0;
    double efficiency = 0.75;
    double sigma_x = 0.006;
    double sigma_y = 0.006;

    double manufactured_peak_rise = 100.0;
};

struct FieldSolution {
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<double> standard_error;
    std::vector<double> mean_steps;
    long long max_steps_hits = 0;
};

struct ErrorSummary {
    double rmse = 0.0;
    double maximum_absolute = 0.0;
};

struct ManufacturedPoisson2D {
    static constexpr bool has_source = true;
    static constexpr bool has_screening = false;
    static constexpr bool has_green_source = true;

    double length_x;
    double length_y;
    double peak_rise;
    wos::SourceMode source_mode = wos::SourceMode::Green;

    bool source_may_intersect([[maybe_unused]] wos::Sphere2D sphere) const {
        return true;
    }

    double exact(wos::Point2D point) const {
        return peak_rise * std::sin(pi * point.x / length_x)
                         * std::sin(pi * point.y / length_y);
    }

    double source(wos::Point2D point) const {
        const double eigenvalue = pi * pi *
            (1.0 / (length_x * length_x) +
             1.0 / (length_y * length_y));
        return eigenvalue * exact(point);
    }

    double boundary([[maybe_unused]] wos::Point2D point,
                    [[maybe_unused]] int boundary_id) const {
        return 0.0;
    }

    double green(wos::Sphere2D sphere, wos::Point2D x,
                 wos::Point2D y) const {
        const double radius = wos::dist(x, y);
        if (!(radius > 0.0)) {
            return 0.0;
        }
        return wos::poisson_green::green_2d(sphere.radius, radius);
    }

    double green_mass(double radius) const {
        return wos::poisson_green::mass_2d(radius);
    }

    wos::Point2D sample_green(wos::Sphere2D sphere, wos::PRNG &rng) const {
        const double radius = wos::poisson_green::sample_radius_2d(
            sphere.radius, rng);
        const double angle = 2.0 * wos::poisson_green::pi * rng.unit();
        return wos::Point2D{
            sphere.centre.x + radius * std::cos(angle),
            sphere.centre.y + radius * std::sin(angle),
        };
    }
};

struct StationaryGaussianHeat2D {
    static constexpr bool has_source = true;
    static constexpr bool has_screening = false;
    static constexpr bool has_green_source = true;

    const welding::GaussianHeatSource2D &heat_source;
    double conductivity;
    wos::SourceMode source_mode = wos::SourceMode::Green;
    double mis_green_probability = 0.5;

    // The Gaussian is normalized on the complete finite plate and therefore
    // has nonzero mathematical support everywhere in the plate.
    bool source_may_intersect([[maybe_unused]] wos::Sphere2D sphere) const {
        return true;
    }

    double source(wos::Point2D point) const {
        return heat_source.volumetric_power_density(point) / conductivity;
    }

    double boundary([[maybe_unused]] wos::Point2D point,
                    [[maybe_unused]] int boundary_id) const {
        return 0.0;
    }

    double green(wos::Sphere2D sphere, wos::Point2D x,
                 wos::Point2D y) const {
        const double radius = wos::dist(x, y);
        if (!(radius > 0.0)) {
            return 0.0;
        }
        return wos::poisson_green::green_2d(sphere.radius, radius);
    }

    double green_mass(double radius) const {
        return wos::poisson_green::mass_2d(radius);
    }

    wos::Point2D sample_green(wos::Sphere2D sphere, wos::PRNG &rng) const {
        const double radius = wos::poisson_green::sample_radius_2d(
            sphere.radius, rng);
        const double angle = 2.0 * wos::poisson_green::pi * rng.unit();
        return wos::Point2D{
            sphere.centre.x + radius * std::cos(angle),
            sphere.centre.y + radius * std::sin(angle),
        };
    }

    wos::Point2D sample_source_proposal(wos::PRNG &rng) const {
        return heat_source.sample_spatial_distribution(rng);
    }

    double source_proposal_pdf(wos::Point2D point) const {
        return heat_source.spatial_probability_density(point);
    }

    double source_mis_green_probability() const {
        return mis_green_probability;
    }
};

std::filesystem::path data_directory() {
    return std::filesystem::path(WELDING_RESULTS_DIR) / "data";
}

void ensure_directory(const std::filesystem::path &directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error(
            "could not create output directory: " + error.message());
    }
}

std::vector<wos::StartPoint<2>> build_start_points(
    const wos::Grid &grid, const wos::BVH<2> &bvh) {
    std::vector<wos::StartPoint<2>> starts(
        static_cast<std::size_t>(grid.Nx) * grid.Ny);
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            starts[index] = wos::get_start_point(bvh, wos::Point2D{x, y});
        }
    }
    return starts;
}

template<typename Equation>
FieldSolution solve_field(const wos::Grid &grid,
                          const wos::BVH<2> &bvh,
                          const std::vector<wos::StartPoint<2>> &starts,
                          const Equation &equation,
                          const Level1Config &config,
                          std::uint64_t case_stream) {
    const std::size_t point_count =
        static_cast<std::size_t>(grid.Nx) * grid.Ny;
    FieldSolution solution{
        std::vector<double>(point_count, 0.0),
        std::vector<double>(point_count, 0.0),
        std::vector<double>(point_count, 0.0),
        std::vector<double>(point_count, 0.0),
        0,
    };

    for (int i = 1; i < grid.Nx - 1; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 1; j < grid.Ny - 1; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            const wos::Point2D point{x, y};

            const std::uint64_t point_seed = wos::splitmix64(
                config.seed ^ case_stream ^
                wos::splitmix64(static_cast<std::uint64_t>(index)));
            wos::PRNG walk_rng(wos::splitmix64(
                point_seed ^ 0x13198A2E03707344ULL));
            wos::PRNG source_rng(wos::splitmix64(
                point_seed ^ 0xA4093822299F31D0ULL));

            const wos::WoSResult result = wos::wos_run(
                bvh, point, starts[index], equation,
                config.walks, config.epsilon,
                walk_rng, source_rng, config.max_steps);
            solution.mean[index] = result.mean;
            solution.variance[index] = result.variance;
            solution.standard_error[index] = result.standard_error;
            solution.mean_steps[index] = result.mean_steps;
            solution.max_steps_hits += result.max_steps_hits;
        }
    }
    return solution;
}

ErrorSummary manufactured_error(const wos::Grid &grid,
                                const FieldSolution &solution,
                                const ManufacturedPoisson2D &equation) {
    double squared_sum = 0.0;
    double maximum_absolute = 0.0;
    std::size_t count = 0;
    for (int i = 1; i < grid.Nx - 1; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 1; j < grid.Ny - 1; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            const double error = solution.mean[index] - equation.exact({x, y});
            squared_sum += error * error;
            maximum_absolute = std::max(maximum_absolute, std::abs(error));
            ++count;
        }
    }
    return ErrorSummary{
        std::sqrt(squared_sum / static_cast<double>(count)),
        maximum_absolute,
    };
}

double mean_interior(const wos::Grid &grid,
                     const std::vector<double> &values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int i = 1; i < grid.Nx - 1; ++i) {
        for (int j = 1; j < grid.Ny - 1; ++j) {
            sum += values[welding::flat_index_2d(i, j, grid.Ny)];
            ++count;
        }
    }
    return sum / static_cast<double>(count);
}

double maximum_interior(const wos::Grid &grid,
                        const std::vector<double> &values) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = 1; i < grid.Nx - 1; ++i) {
        for (int j = 1; j < grid.Ny - 1; ++j) {
            maximum = std::max(
                maximum, values[welding::flat_index_2d(i, j, grid.Ny)]);
        }
    }
    return maximum;
}

void write_manufactured_outputs(const wos::Grid &grid,
                                const Level1Config &config,
                                const ManufacturedPoisson2D &equation,
                                const FieldSolution &solution,
                                double solve_seconds) {
    const ErrorSummary errors = manufactured_error(grid, solution, equation);
    const int centre_i = grid.Nx / 2;
    const int centre_j = grid.Ny / 2;
    const std::size_t centre_index = welding::flat_index_2d(
        centre_i, centre_j, grid.Ny);

    const std::filesystem::path summary_path =
        data_directory() / "level1_manufactured_summary.csv";
    std::ofstream summary(summary_path);
    if (!summary) {
        throw std::runtime_error("could not open " + summary_path.string());
    }
    summary << std::setprecision(17);
    summary << "length_x,length_y,grid_x,grid_y,walks,epsilon,seed,"
               "peak_rise,center_mean,center_standard_error,center_exact,"
               "center_absolute_error,field_rmse,field_max_absolute_error,"
               "mean_standard_error,max_standard_error,mean_steps,max_steps_hits,"
               "solve_seconds\n";
    summary << config.length_x << ',' << config.length_y << ','
            << grid.Nx << ',' << grid.Ny << ',' << config.walks << ','
            << config.epsilon << ',' << config.seed << ','
            << config.manufactured_peak_rise << ','
            << solution.mean[centre_index] << ','
            << solution.standard_error[centre_index] << ','
            << config.manufactured_peak_rise << ','
            << std::abs(solution.mean[centre_index] -
                        config.manufactured_peak_rise) << ','
            << errors.rmse << ',' << errors.maximum_absolute << ','
            << mean_interior(grid, solution.standard_error) << ','
            << maximum_interior(grid, solution.standard_error) << ','
            << mean_interior(grid, solution.mean_steps) << ','
            << solution.max_steps_hits << ',' << solve_seconds << '\n';

    const std::filesystem::path field_path =
        data_directory() / "level1_manufactured_field.csv";
    std::ofstream field(field_path);
    if (!field) {
        throw std::runtime_error("could not open " + field_path.string());
    }
    field << std::setprecision(17);
    field << "x,y,temperature_rise,standard_error,variance,mean_steps,"
             "source,exact,error,absolute_error\n";
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            const double exact = equation.exact({x, y});
            const double error = solution.mean[index] - exact;
            field << x << ',' << y << ',' << solution.mean[index] << ','
                  << solution.standard_error[index] << ','
                  << solution.variance[index] << ','
                  << solution.mean_steps[index] << ','
                  << equation.source({x, y}) << ',' << exact << ','
                  << error << ',' << std::abs(error) << '\n';
        }
    }

    std::printf("Level 1A: manufactured steady Poisson\n");
    std::printf("  centre rise:          %.9g K (exact %.9g K)\n",
                solution.mean[centre_index], config.manufactured_peak_rise);
    std::printf("  centre standard err:  %.6g K\n",
                solution.standard_error[centre_index]);
    std::printf("  field RMSE:           %.6g K\n", errors.rmse);
    std::printf("  field max abs error:  %.6g K\n", errors.maximum_absolute);
    std::printf("  solve time:           %.6f s\n\n", solve_seconds);
}

struct PeakLocation {
    double value;
    double x;
    double y;
    std::size_t index;
};

PeakLocation find_peak(const wos::Grid &grid,
                       const std::vector<double> &values) {
    PeakLocation peak{-std::numeric_limits<double>::infinity(), 0.0, 0.0, 0};
    for (int i = 1; i < grid.Nx - 1; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 1; j < grid.Ny - 1; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            if (values[index] > peak.value) {
                peak = PeakLocation{values[index], x, y, index};
            }
        }
    }
    return peak;
}

struct SymmetrySummary {
    double maximum_relative;
    double rms_relative;
    double within_three_standard_errors_fraction;
};

SymmetrySummary symmetry_summary(const wos::Grid &grid,
                                 const std::vector<double> &values,
                                 const std::vector<double> &standard_error,
                                 double peak_rise) {
    double maximum_difference = 0.0;
    double squared_difference_sum = 0.0;
    std::size_t comparison_count = 0;
    std::size_t within_three_standard_errors = 0;
    for (int i = 0; i < grid.Nx; ++i) {
        for (int j = 0; j < grid.Ny; ++j) {
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            const std::size_t mirror_x = welding::flat_index_2d(
                grid.Nx - 1 - i, j, grid.Ny);
            const std::size_t mirror_y = welding::flat_index_2d(
                i, grid.Ny - 1 - j, grid.Ny);
            for (const std::size_t mirror : {mirror_x, mirror_y}) {
                const double difference = std::abs(
                    values[index] - values[mirror]);
                const double combined_standard_error = std::hypot(
                    standard_error[index], standard_error[mirror]);
                maximum_difference = std::max(maximum_difference, difference);
                squared_difference_sum += difference * difference;
                ++comparison_count;
                if (difference <= 3.0 * combined_standard_error ||
                    (difference == 0.0 && combined_standard_error == 0.0)) {
                    ++within_three_standard_errors;
                }
            }
        }
    }
    return SymmetrySummary{
        maximum_difference / peak_rise,
        std::sqrt(squared_difference_sum /
                  static_cast<double>(comparison_count)) / peak_rise,
        static_cast<double>(within_three_standard_errors) /
            static_cast<double>(comparison_count),
    };
}

void write_gaussian_outputs(const wos::Grid &grid,
                            const Level1Config &config,
                            const welding::GaussianHeatSource2D &heat_source,
                            const StationaryGaussianHeat2D &equation,
                            const FieldSolution &solution,
                            double solve_seconds,
                            const std::string &file_stem,
                            const char *sampling_label) {
    const PeakLocation peak = find_peak(grid, solution.mean);
    const double integrated_power = heat_source.integrated_power_midpoint(800, 600);
    const double relative_power_error = std::abs(
        integrated_power - heat_source.absorbed_power()) /
        heat_source.absorbed_power();
    const SymmetrySummary symmetry = symmetry_summary(
        grid, solution.mean, solution.standard_error, peak.value);

    const std::filesystem::path summary_path =
        data_directory() / (file_stem + "_summary.csv");
    std::ofstream summary(summary_path);
    if (!summary) {
        throw std::runtime_error("could not open " + summary_path.string());
    }
    summary << std::setprecision(17);
    summary << "source_mode,mis_green_probability,length_x,length_y,thickness,"
               "conductivity,ambient_temperature,"
               "electrical_power,efficiency,absorbed_power,source_x,source_y,"
               "sigma_x,sigma_y,gaussian_integral_x,gaussian_integral_y,"
               "peak_power_density,integrated_power,power_relative_error,"
               "grid_x,grid_y,walks,epsilon,seed,peak_temperature_rise,"
               "peak_temperature,peak_x,peak_y,peak_standard_error,"
               "symmetry_max_relative_residual,symmetry_rms_relative_residual,"
               "symmetry_within_three_standard_errors_fraction,mean_standard_error,"
               "max_standard_error,mean_steps,max_steps_hits,solve_seconds\n";
    summary << static_cast<int>(equation.source_mode) << ','
            << equation.mis_green_probability << ','
            << config.length_x << ',' << config.length_y << ','
            << config.thickness << ',' << config.conductivity << ','
            << config.ambient_temperature << ',' << config.electrical_power << ','
            << config.efficiency << ',' << heat_source.absorbed_power() << ','
            << heat_source.centre_x() << ',' << heat_source.centre_y() << ','
            << heat_source.sigma_x() << ',' << heat_source.sigma_y() << ','
            << heat_source.integral_x() << ',' << heat_source.integral_y() << ','
            << heat_source.peak_power_density() << ',' << integrated_power << ','
            << relative_power_error << ',' << grid.Nx << ',' << grid.Ny << ','
            << config.walks << ',' << config.epsilon << ',' << config.seed << ','
            << peak.value << ',' << peak.value + config.ambient_temperature << ','
            << peak.x << ',' << peak.y << ','
            << solution.standard_error[peak.index] << ','
            << symmetry.maximum_relative << ',' << symmetry.rms_relative << ','
            << symmetry.within_three_standard_errors_fraction << ','
            << mean_interior(grid, solution.standard_error) << ','
            << maximum_interior(grid, solution.standard_error) << ','
            << mean_interior(grid, solution.mean_steps) << ','
            << solution.max_steps_hits << ',' << solve_seconds << '\n';

    const std::filesystem::path field_path =
        data_directory() / (file_stem + "_field.csv");
    const std::filesystem::path centerlines_path =
        data_directory() / (file_stem + "_centerlines.csv");
    std::ofstream field(field_path);
    std::ofstream centerlines(centerlines_path);
    if (!field || !centerlines) {
        throw std::runtime_error("could not open Gaussian output CSV files");
    }
    field << std::setprecision(17);
    centerlines << std::setprecision(17);
    field << "x,y,temperature_rise,temperature,standard_error,variance,"
             "mean_steps,volumetric_power_density,poisson_source\n";
    centerlines << "axis,coordinate,x,y,temperature_rise,temperature,"
                   "standard_error\n";

    const int centre_i = grid.Nx / 2;
    const int centre_j = grid.Ny / 2;
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(
            grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(
                grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
            const wos::Point2D point{x, y};
            field << x << ',' << y << ',' << solution.mean[index] << ','
                  << solution.mean[index] + config.ambient_temperature << ','
                  << solution.standard_error[index] << ','
                  << solution.variance[index] << ','
                  << solution.mean_steps[index] << ','
                  << heat_source.volumetric_power_density(point) << ','
                  << equation.source(point) << '\n';

            if (j == centre_j) {
                centerlines << "x," << x << ',' << x << ',' << y << ','
                            << solution.mean[index] << ','
                            << solution.mean[index] + config.ambient_temperature
                            << ',' << solution.standard_error[index] << '\n';
            }
            if (i == centre_i) {
                centerlines << "y," << y << ',' << x << ',' << y << ','
                            << solution.mean[index] << ','
                            << solution.mean[index] + config.ambient_temperature
                            << ',' << solution.standard_error[index] << '\n';
            }
        }
    }

    std::printf("Level 1B: stationary finite-domain Gaussian heat source (%s)\n",
                sampling_label);
    std::printf("  absorbed power:       %.9g W\n", heat_source.absorbed_power());
    std::printf("  integrated power:     %.9g W\n", integrated_power);
    std::printf("  relative power error: %.6g\n", relative_power_error);
    std::printf("  peak temperature:     %.9g K at (%.6g, %.6g) m\n",
                peak.value + config.ambient_temperature, peak.x, peak.y);
    std::printf("  peak standard error:  %.6g K\n",
                solution.standard_error[peak.index]);
    std::printf("  symmetry max residual:%.6g\n", symmetry.maximum_relative);
    std::printf("  symmetry RMS residual:%.6g\n", symmetry.rms_relative);
    std::printf("  symmetry within 3 SE: %.4f%%\n",
                100.0 * symmetry.within_three_standard_errors_fraction);
    std::printf("  max-step hits:        %lld\n", solution.max_steps_hits);
    std::printf("  solve time:           %.6f s\n\n", solve_seconds);
}

int run_level1() {
    const Level1Config config;
    const wos::Grid grid{
        config.nx, config.ny, 1,
        0.0, config.length_x,
        0.0, config.length_y,
        0.0, 0.0,
    };

    ensure_directory(data_directory());
    const wos::Mesh<2> mesh = welding::make_rectangular_plate(
        grid.xmin, grid.xmax, grid.ymin, grid.ymax);
    const std::unique_ptr<wos::BVH<2>> bvh = wos::build_bvh(mesh);
    const std::vector<wos::StartPoint<2>> starts =
        build_start_points(grid, *bvh);

    std::printf("Level-1 stationary welding heat verification\n");
    std::printf("  grid:                  %d x %d\n", grid.Nx, grid.Ny);
    std::printf("  plate:                 %.6g x %.6g x %.6g m\n",
                config.length_x, config.length_y, config.thickness);
    std::printf("  walks/point:           %d\n", config.walks);
    std::printf("  epsilon:               %.9g m\n", config.epsilon);
    std::printf("  seed:                  %llu\n\n",
                static_cast<unsigned long long>(config.seed));

    const ManufacturedPoisson2D manufactured{
        config.length_x,
        config.length_y,
        config.manufactured_peak_rise,
    };
    const auto manufactured_start = std::chrono::steady_clock::now();
    const FieldSolution manufactured_solution = solve_field(
        grid, *bvh, starts, manufactured, config,
        0x4C4556454C31414FULL);
    const double manufactured_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - manufactured_start).count();
    write_manufactured_outputs(
        grid, config, manufactured, manufactured_solution,
        manufactured_seconds);

    const welding::GaussianHeatSource2D heat_source(
        grid.xmin, grid.xmax, grid.ymin, grid.ymax,
        config.thickness,
        config.electrical_power, config.efficiency,
        0.5 * config.length_x, 0.5 * config.length_y,
        config.sigma_x, config.sigma_y);
    const StationaryGaussianHeat2D gaussian_green{
        heat_source,
        config.conductivity,
        wos::SourceMode::Green,
        0.5,
    };
    const auto gaussian_green_start = std::chrono::steady_clock::now();
    const FieldSolution gaussian_green_solution = solve_field(
        grid, *bvh, starts, gaussian_green, config,
        0x4C4556454C314247ULL);
    const double gaussian_green_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gaussian_green_start).count();
    write_gaussian_outputs(
        grid, config, heat_source, gaussian_green, gaussian_green_solution,
        gaussian_green_seconds, "level1_gaussian", "Green baseline");

    // Use the same case stream as the baseline. Walk and source RNGs are
    // independent, so both methods see identical WoS paths while their source
    // draws follow their respective proposals.
    const StationaryGaussianHeat2D gaussian_mis{
        heat_source,
        config.conductivity,
        wos::SourceMode::GreenSourceMIS,
        0.5,
    };
    const auto gaussian_mis_start = std::chrono::steady_clock::now();
    const FieldSolution gaussian_mis_solution = solve_field(
        grid, *bvh, starts, gaussian_mis, config,
        0x4C4556454C314247ULL);
    const double gaussian_mis_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gaussian_mis_start).count();
    write_gaussian_outputs(
        grid, config, heat_source, gaussian_mis, gaussian_mis_solution,
        gaussian_mis_seconds, "level1_gaussian_mis", "Green-source MIS");

    std::printf("Finished. Results are in %s\n", data_directory().string().c_str());
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int status = 0;
    if (size != 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "wos_welding_level1 currently supports one MPI rank; got %d\n",
                         size);
        }
        status = 1;
    } else if (argc != 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "wos_welding_level1 currently uses its built-in validated "
                         "case and accepts no command-line options\n");
        }
        status = 1;
    } else {
        try {
            status = run_level1();
        } catch (const std::exception &error) {
            std::fprintf(stderr, "wos_welding_level1: %s\n", error.what());
            status = 1;
        }
    }

    MPI_Finalize();
    return status;
}
