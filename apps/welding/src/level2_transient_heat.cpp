#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#ifdef WELDING_LEVEL2_HAS_OPENMP
#include <omp.h>
#endif

#include "welding/cli.hpp"
#include "welding/file_output.hpp"
#include "welding/field_sampler.hpp"
#include "welding/gaussian_heat_source.hpp"
#include "welding/grid_utils.hpp"
#include "welding/random_streams.hpp"
#include "welding/rectangular_plate.hpp"
#include "welding/screened_heat.hpp"
#include "welding/temperature_field.hpp"
#include "wos/geometry/fcpw_scene.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/mesh.hpp"
#include "wos/solver/wos.hpp"

namespace {

#ifndef WELDING_RESULTS_DIR
#define WELDING_RESULTS_DIR "apps/welding/results"
#endif

struct Level2Config {
    int nx = 49;
    int ny = 33;
    int walks = 500;
    int max_steps = 1'000;
    int field_stride = 1;
    int threads = 0;
    welding::FieldInterpolation2D history_interpolation =
        welding::FieldInterpolation2D::Cubic;
    double dt = 0.1;
    double total_time = 10.0;
    double heat_on_time = 3.0;
    double epsilon = 8e-6;
    std::uint64_t seed = 20260819ULL;

    double length_x = 0.12;
    double length_y = 0.08;
    double thickness = 0.006;
    double conductivity = 35.0;
    double density = 7'800.0;
    double specific_heat = 600.0;
    double ambient_temperature = 293.15;

    double electrical_power = 800.0;
    double efficiency = 0.75;
    double sigma_x = 0.006;
    double sigma_y = 0.006;

    std::string output_prefix =
        WELDING_RESULTS_DIR "/data/level2_transient_heat";
};

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Level-2 transient welding-temperature baseline on a rectangular\n"
        "thin plate. A stationary Gaussian volumetric heat source is active\n"
        "for 3 seconds by default, followed by cooling. Each implicit-Euler\n"
        "step is solved with screened walk on spheres. This executable\n"
        "currently supports exactly one MPI rank.\n"
        "\n"
        "Options:\n"
        "  --nx N                grid points in x (default: 49)\n"
        "  --ny N                grid points in y (default: 33)\n"
        "  --dt X                time-step size in seconds (default: 0.1)\n"
        "  --total-time X        final time in seconds (default: 10)\n"
        "  --heat-on-time X      heating duration in seconds (default: 3)\n"
        "  --walks N             WoS paths per interior point (default: 500)\n"
        "  --epsilon X           boundary stopping distance (default: 8e-6)\n"
        "  --max-steps N         maximum sphere steps per path (default: 1000)\n"
        "  --field-stride N      write every Nth full field (default: 1)\n"
        "  --history-interpolation MODE\n"
        "                        bilinear or cubic (default: cubic)\n"
        "  --threads N           OpenMP threads (default: runtime setting)\n"
        "  --seed N              reproducible unsigned seed\n"
        "  --output-prefix PATH  output path without suffix\n"
        "  -h, --help            show this message\n",
        program);
}

Level2Config parse_options(int argc, char **argv) {
    Level2Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value_after = [&](const char *name) {
            return welding::option_value(i, argc, argv, name);
        };

        if (argument == "--nx") {
            config.nx = welding::parse_positive_int(value_after("--nx"), "nx");
        } else if (argument == "--ny") {
            config.ny = welding::parse_positive_int(value_after("--ny"), "ny");
        } else if (argument == "--dt") {
            config.dt = welding::parse_positive_double(
                value_after("--dt"), "time step");
        } else if (argument == "--total-time") {
            config.total_time = welding::parse_positive_double(
                value_after("--total-time"), "total time");
        } else if (argument == "--heat-on-time") {
            config.heat_on_time = welding::parse_positive_double(
                value_after("--heat-on-time"), "heating duration");
        } else if (argument == "--walks") {
            config.walks = welding::parse_positive_int(
                value_after("--walks"), "walk count");
        } else if (argument == "--epsilon") {
            config.epsilon = welding::parse_positive_double(
                value_after("--epsilon"), "epsilon");
        } else if (argument == "--max-steps") {
            config.max_steps = welding::parse_positive_int(
                value_after("--max-steps"), "maximum step count");
        } else if (argument == "--field-stride") {
            config.field_stride = welding::parse_positive_int(
                value_after("--field-stride"), "field stride");
        } else if (argument == "--history-interpolation") {
            const std::string mode = value_after("--history-interpolation");
            if (mode == "bilinear") {
                config.history_interpolation =
                    welding::FieldInterpolation2D::Bilinear;
            } else if (mode == "cubic") {
                config.history_interpolation =
                    welding::FieldInterpolation2D::Cubic;
            } else {
                throw std::invalid_argument(
                    "--history-interpolation must be bilinear or cubic");
            }
        } else if (argument == "--threads") {
            config.threads = welding::parse_positive_int(
                value_after("--threads"), "thread count");
        } else if (argument == "--seed") {
            config.seed = welding::parse_uint64(value_after("--seed"));
        } else if (argument == "--output-prefix") {
            config.output_prefix = value_after("--output-prefix");
            if (config.output_prefix.empty()) {
                throw std::invalid_argument(
                    "--output-prefix requires a non-empty path");
            }
        } else if (argument == "-h" || argument == "--help") {
            continue;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (config.nx < 3 || config.ny < 3 ||
        config.nx % 2 == 0 || config.ny % 2 == 0) {
        throw std::invalid_argument(
            "--nx and --ny must be odd integers of at least 3");
    }
    if (!(config.heat_on_time < config.total_time)) {
        throw std::invalid_argument(
            "--heat-on-time must be smaller than --total-time");
    }
    const double step_count = config.total_time / config.dt;
    if (std::abs(step_count - std::round(step_count)) >
        1e-10 * std::max(1.0, step_count)) {
        throw std::invalid_argument(
            "--total-time must be an integer multiple of --dt");
    }
    if (!(config.epsilon < 0.5 * std::min(config.length_x, config.length_y))) {
        throw std::invalid_argument("--epsilon is too large for the plate");
    }
    return config;
}

double heating_fraction(double time_begin, double time_end,
                        double heat_on_time) {
    const double overlap_begin = std::max(0.0, time_begin);
    const double overlap_end = std::min(time_end, heat_on_time);
    if (!(overlap_end > overlap_begin)) {
        return 0.0;
    }
    return (overlap_end - overlap_begin) / (time_end - time_begin);
}

struct TransientGaussianHeat2D
    : welding::ZeroDirichletScreenedHeat2D {
    double power_fraction;
    double conductivity;
    const welding::FieldSampler2D &previous;
    const welding::GaussianHeatSource2D &heat_source;

    TransientGaussianHeat2D(
            double alpha, double source_power_fraction,
            double thermal_conductivity,
            const welding::FieldSampler2D &previous_field,
            const welding::GaussianHeatSource2D &source,
            wos::SourceMode mode = wos::SourceMode::Green)
        : ZeroDirichletScreenedHeat2D(alpha),
          power_fraction(source_power_fraction),
          conductivity(thermal_conductivity),
          previous(previous_field), heat_source(source) {
        source_mode = mode;
    }

    double source(wos::Point2D point) const {
        const double previous_contribution =
            screening_parameter_squared() *
            std::max(0.0, previous.sample(point));
        const double heat_contribution = power_fraction > 0.0
            ? power_fraction * heat_source.volumetric_power_density(point)
                / conductivity
            : 0.0;
        return previous_contribution + heat_contribution;
    }
};

int run_level2(const Level2Config &config) {
#ifdef WELDING_LEVEL2_HAS_OPENMP
    if (config.threads > 0) {
        omp_set_num_threads(config.threads);
    }
    const int effective_threads = omp_get_max_threads();
#else
    if (config.threads > 1) {
        throw std::invalid_argument(
            "--threads greater than 1 requires an OpenMP-enabled build");
    }
    const int effective_threads = 1;
#endif
    const double xmin = -0.5 * config.length_x;
    const double xmax = 0.5 * config.length_x;
    const double ymin = -0.5 * config.length_y;
    const double ymax = 0.5 * config.length_y;
    const wos::Grid grid{
        config.nx, config.ny, 1,
        xmin, xmax, ymin, ymax, 0.0, 0.0,
    };
    const std::size_t point_count =
        static_cast<std::size_t>(grid.Nx) * grid.Ny;
    const int step_count = static_cast<int>(std::llround(
        config.total_time / config.dt));
    const double diffusivity = config.conductivity /
        (config.density * config.specific_heat);
    const double alpha = 1.0 / std::sqrt(diffusivity * config.dt);
    const wos::solver::WoS solver(
        {config.walks, config.epsilon, config.max_steps});

    wos::Mesh<2> mesh = welding::make_rectangular_plate(
        xmin, xmax, ymin, ymax);
    wos::FcpwGeometryScene<2> geometry_scene(mesh);
    const std::vector<wos::solver::StartPoint<2>> starts =
        welding::build_start_points_2d(grid, geometry_scene);

    const welding::GaussianHeatSource2D heat_source(
        xmin, xmax, ymin, ymax,
        config.thickness,
        config.electrical_power,
        config.efficiency,
        0.0, 0.0,
        config.sigma_x, config.sigma_y);

    std::vector<double> previous(point_count, 0.0);
    std::vector<double> next(point_count, 0.0);
    std::vector<double> standard_error(point_count, 0.0);

    const std::filesystem::path prefix(config.output_prefix);
    const std::filesystem::path summary_path =
        prefix.string() + "_summary.csv";
    const std::filesystem::path history_path =
        prefix.string() + "_history.csv";
    const std::filesystem::path metadata_path =
        prefix.string() + "_metadata.csv";
    welding::ensure_parent_directory(summary_path);
    welding::ensure_parent_directory(history_path);
    welding::ensure_parent_directory(metadata_path);
    std::ofstream summary_output(summary_path);
    std::ofstream history_output(history_path);
    std::ofstream metadata_output(metadata_path);
    if (!summary_output || !history_output || !metadata_output) {
        throw std::runtime_error("could not open Level-2 output files");
    }
    summary_output << std::setprecision(17);
    history_output << std::setprecision(17);
    metadata_output << std::setprecision(17);

    metadata_output << "key,value,unit\n"
                    << "nx," << config.nx << ",1\n"
                    << "ny," << config.ny << ",1\n"
                    << "walks," << config.walks << ",1\n"
                    << "max_steps," << config.max_steps << ",1\n"
                    << "field_stride," << config.field_stride << ",1\n"
                    << "history_interpolation,"
                    << welding::field_interpolation_name(
                           config.history_interpolation)
                    << ",1\n"
                    << "threads," << effective_threads << ",1\n"
                    << "dt," << config.dt << ",s\n"
                    << "total_time," << config.total_time << ",s\n"
                    << "heat_on_time," << config.heat_on_time << ",s\n"
                    << "epsilon," << config.epsilon << ",m\n"
                    << "seed," << config.seed << ",1\n"
                    << "length_x," << config.length_x << ",m\n"
                    << "length_y," << config.length_y << ",m\n"
                    << "thickness," << config.thickness << ",m\n"
                    << "conductivity," << config.conductivity << ",W/(m K)\n"
                    << "density," << config.density << ",kg/m3\n"
                    << "specific_heat," << config.specific_heat << ",J/(kg K)\n"
                    << "ambient_temperature," << config.ambient_temperature << ",K\n"
                    << "electrical_power," << config.electrical_power << ",W\n"
                    << "efficiency," << config.efficiency << ",1\n"
                    << "absorbed_power," << heat_source.absorbed_power() << ",W\n"
                    << "sigma_x," << config.sigma_x << ",m\n"
                    << "sigma_y," << config.sigma_y << ",m\n"
                    << "diffusivity," << diffusivity << ",m2/s\n"
                    << "alpha," << alpha << ",1/m\n"
                    << "source_mode,screened_green,1\n"
                    << "nonnegative_history_clamp,true,1\n"
                    << "boundary_condition,zero_temperature_rise,1\n";

    summary_output
        << "step,time,source_fraction,absorbed_power,cumulative_input_energy,"
           "center_temperature_rise,center_temperature,center_standard_error,"
           "peak_temperature_rise,peak_temperature,peak_x,peak_y,"
           "mean_standard_error,max_standard_error,"
           "minimum_interior_temperature_rise,negative_interior_points,"
           "thermal_energy_rise,"
           "max_steps_hits,step_seconds\n";
    history_output
        << "step,time,x,y,temperature_rise,temperature,standard_error\n";

    const int center_i = grid.Nx / 2;
    const int center_j = grid.Ny / 2;
    const std::size_t center_index = welding::flat_index_2d(
        center_i, center_j, grid.Ny);
    welding::write_temperature_field_2d(
        history_output, 0, 0.0, config.ambient_temperature,
        grid, previous, standard_error);
    summary_output
        << "0,0,0,0,0,0," << config.ambient_temperature
        << ",0,0," << config.ambient_temperature
        << ",0,0,0,0,0,0,0,0,0\n";

    std::printf("Level-2 stationary Gaussian transient heating and cooling\n");
    std::printf("  plate:          %.6g x %.6g x %.6g m\n",
                config.length_x, config.length_y, config.thickness);
    std::printf("  grid:           %d x %d\n", grid.Nx, grid.Ny);
    std::printf("  rho, cp, k:     %.6g kg/m3, %.6g J/(kg K), %.6g W/(m K)\n",
                config.density, config.specific_heat, config.conductivity);
    std::printf("  diffusivity:    %.12g m2/s\n", diffusivity);
    std::printf("  dt / steps:     %.6g s / %d\n", config.dt, step_count);
    std::printf("  heating:        0 to %.6g s\n", config.heat_on_time);
    std::printf("  absorbed power: %.6g W\n", heat_source.absorbed_power());
    std::printf("  alpha:          %.12g 1/m\n", alpha);
    std::printf("  walks/point:    %d\n", config.walks);
    std::printf("  history interp: %s\n",
                welding::field_interpolation_name(
                    config.history_interpolation));
    std::printf("  threads:        %d\n", effective_threads);
    std::printf("  source mode:    screened Green\n");
    std::printf("  summary:        %s\n", summary_path.string().c_str());
    std::printf("  history:        %s\n", history_path.string().c_str());
    std::printf("  metadata:       %s\n\n", metadata_path.string().c_str());
    std::printf(
        " step      time  source     center rise       peak rise"
        "      thermal energy    step seconds\n");

    double cumulative_input_energy = 0.0;
    for (int step = 1; step <= step_count; ++step) {
        const double time_begin = (step - 1) * config.dt;
        const double time_end = step * config.dt;
        const double source_fraction = heating_fraction(
            time_begin, time_end, config.heat_on_time);
        cumulative_input_energy += source_fraction
            * heat_source.absorbed_power() * config.dt;

        welding::FieldSampler2D sampler(
            grid, previous, config.history_interpolation);
        const TransientGaussianHeat2D equation{
            alpha,
            source_fraction,
            config.conductivity,
            sampler,
            heat_source,
            wos::SourceMode::Green,
        };

        std::fill(next.begin(), next.end(), 0.0);
        std::fill(standard_error.begin(), standard_error.end(), 0.0);
        long long max_steps_hits = 0;
        const auto step_start = std::chrono::steady_clock::now();

#ifdef WELDING_LEVEL2_HAS_OPENMP
#pragma omp parallel for collapse(2) reduction(+ : max_steps_hits) schedule(static)
#endif
        for (int i = 1; i < grid.Nx - 1; ++i) {
            for (int j = 1; j < grid.Ny - 1; ++j) {
                const double x = wos::grid_coordinate(
                    grid.xmin, grid.xmax, i, grid.Nx);
                const double y = wos::grid_coordinate(
                    grid.ymin, grid.ymax, j, grid.Ny);
                const std::size_t index = welding::flat_index_2d(
                    i, j, grid.Ny);
                const wos::Point2D point{x, y};

                welding::PathRandomStreams random =
                    welding::make_path_random_streams(welding::point_seed(
                        config.seed,
                        wos::splitmix64(static_cast<std::uint64_t>(step)),
                        index));

                const wos::solver::Result result = solver.solve(
                    geometry_scene, point, starts[index], equation,
                    random.walk, random.source);
                next[index] = result.mean;
                standard_error[index] = result.standard_error;
                max_steps_hits += result.max_steps_hits;
            }
        }

        const double step_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        const welding::TemperatureFieldSummary2D field_summary =
            welding::summarize_temperature_field_2d(
                grid, next, standard_error, config.density,
                config.specific_heat, config.thickness);
        const double peak_x = wos::grid_coordinate(
            grid.xmin, grid.xmax, field_summary.peak_i, grid.Nx);
        const double peak_y = wos::grid_coordinate(
            grid.ymin, grid.ymax, field_summary.peak_j, grid.Ny);

        summary_output
            << step << ',' << time_end << ',' << source_fraction << ','
            << source_fraction * heat_source.absorbed_power() << ','
            << cumulative_input_energy << ','
            << next[center_index] << ','
            << config.ambient_temperature + next[center_index] << ','
            << standard_error[center_index] << ','
            << field_summary.peak_rise << ','
            << config.ambient_temperature + field_summary.peak_rise << ','
            << peak_x << ',' << peak_y << ','
            << field_summary.mean_standard_error << ','
            << field_summary.max_standard_error << ','
            << field_summary.minimum_interior_rise << ','
            << field_summary.negative_interior_points << ','
            << field_summary.thermal_energy_rise << ','
            << max_steps_hits << ',' << step_seconds << '\n';

        if (step % config.field_stride == 0 ||
            step == step_count ||
            (time_begin < config.heat_on_time &&
             time_end >= config.heat_on_time)) {
            welding::write_temperature_field_2d(
                history_output, step, time_end,
                config.ambient_temperature, grid, next, standard_error);
        }

        const int report_stride = std::max(1, step_count / 20);
        if (step == 1 || step == step_count ||
            step % report_stride == 0 ||
            (time_begin < config.heat_on_time &&
             time_end >= config.heat_on_time)) {
            std::printf("%5d %9.4f %7.3f %15.6g %15.6g %19.6g %15.6g\n",
                        step, time_end, source_fraction,
                        next[center_index], field_summary.peak_rise,
                        field_summary.thermal_energy_rise, step_seconds);
        }

        previous.swap(next);
    }

    const double expected_input_energy =
        heat_source.absorbed_power() * config.heat_on_time;
    const double energy_error = cumulative_input_energy - expected_input_energy;
    std::printf("\nFinished Level 2.\n");
    std::printf("  cumulative source energy: %.12g J\n",
                cumulative_input_energy);
    std::printf("  expected source energy:   %.12g J\n",
                expected_input_energy);
    std::printf("  source-energy error:      %.12g J\n", energy_error);
    std::printf("  final center rise:        %.12g K\n",
                previous[center_index]);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int status = 0;
    if (size != 1) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "wos_welding_level2 currently supports one MPI rank; got %d\n",
                         size);
        }
        status = 1;
    } else {
        try {
            status = run_level2(parse_options(argc, argv));
        } catch (const std::exception &error) {
            std::fprintf(stderr, "wos_welding_level2: %s\n", error.what());
            status = 1;
        }
    }

    MPI_Finalize();
    return status;
}
