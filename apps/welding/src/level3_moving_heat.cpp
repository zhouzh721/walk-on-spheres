#include <algorithm>
#include <array>
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

#ifdef WELDING_LEVEL3_HAS_OPENMP
#include <omp.h>
#endif

#include "welding/cli.hpp"
#include "welding/file_output.hpp"
#include "welding/field_sampler.hpp"
#include "welding/grid_utils.hpp"
#include "welding/moving_gaussian_heat_source.hpp"
#include "welding/random_streams.hpp"
#include "welding/rectangular_plate.hpp"
#include "welding/screened_heat.hpp"
#include "welding/temperature_field.hpp"
#include "wos/geometry/fcpw_scene.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/solver/wos.hpp"

namespace {

#ifndef WELDING_RESULTS_DIR
#define WELDING_RESULTS_DIR "apps/welding/results"
#endif

struct Level3Config {
    // Fixed physical and time-discretization baseline.
    static constexpr int nx = 97;
    static constexpr int ny = 65;
    static constexpr int max_steps = 1'000;
    static constexpr double dt = 0.1;
    static constexpr double total_time = 15.0;
    static constexpr double heat_off_time = 5.0;
    static constexpr double epsilon = 8e-6;
    static constexpr double length_x = 0.12;
    static constexpr double length_y = 0.08;
    static constexpr double thickness = 0.006;
    static constexpr double conductivity = 35.0;
    static constexpr double density = 7'800.0;
    static constexpr double specific_heat = 600.0;
    static constexpr double ambient_temperature = 293.15;
    static constexpr double electrical_power = 800.0;
    static constexpr double efficiency = 0.75;
    static constexpr double sigma_x = 0.006;
    static constexpr double sigma_y = 0.006;
    static constexpr double source_start_x = -0.05;
    static constexpr double source_end_x = 0.05;
    static constexpr double weld_y = 0.0;
    static constexpr double weld_speed = 0.02;

    // Numerical/runtime controls remain configurable for validation runs.
    int walks = 500;
    int field_stride = 1;
    int threads = 0;
    std::uint64_t seed = 20260819ULL;
    std::string output_prefix =
        WELDING_RESULTS_DIR "/data/level3_moving_heat";
};

struct Probe {
    const char *name;
    double x;
    double y;
    double source_arrival_time;
};

constexpr std::array<Probe, 7> probes{{
    {"centreline_m20", -0.020, 0.000, 1.5},
    {"centreline_m10", -0.010, 0.000, 2.0},
    {"centreline_0",    0.000, 0.000, 2.5},
    {"centreline_p10",  0.010, 0.000, 3.0},
    {"centreline_p20",  0.020, 0.000, 3.5},
    {"transverse_p5",   0.000, 0.005, 2.5},
    {"transverse_p10",  0.000, 0.010, 2.5},
}};

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [runtime options]\n\n"
        "Level-3 fixed moving-source welding baseline.  A 600 W absorbed\n"
        "Gaussian source moves from x=-50 mm to x=+50 mm at 20 mm/s during\n"
        "0--5 s, then switches off while the plate cools until 15 s.\n"
        "Physical parameters, geometry, trajectory, dt and grid are fixed.\n"
        "The executable currently supports exactly one MPI rank.\n\n"
        "Runtime options:\n"
        "  --walks N             WoS paths per interior point (default: 500)\n"
        "  --field-stride N      write every Nth complete field (default: 1)\n"
        "  --threads N           OpenMP threads (default: runtime setting)\n"
        "  --seed N              reproducible unsigned seed\n"
        "  --output-prefix PATH  output path without suffix\n"
        "  -h, --help            show this message\n",
        program);
}

Level3Config parse_options(int argc, char **argv) {
    Level3Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value_after = [&](const char *name) {
            return welding::option_value(i, argc, argv, name);
        };
        if (argument == "--walks") {
            config.walks = welding::parse_positive_int(
                value_after("--walks"), "walk count");
        } else if (argument == "--field-stride") {
            config.field_stride = welding::parse_positive_int(
                value_after("--field-stride"), "field stride");
        } else if (argument == "--threads") {
            config.threads = welding::parse_positive_int(
                value_after("--threads"), "thread count");
        } else if (argument == "--seed") {
            config.seed = welding::parse_uint64(value_after("--seed"));
        } else if (argument == "--output-prefix") {
            config.output_prefix = value_after("--output-prefix");
            if (config.output_prefix.empty()) {
                throw std::invalid_argument("empty output prefix");
            }
        } else if (argument == "-h" || argument == "--help") {
            continue;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return config;
}

double source_x_at(double time) {
    const double active_time = std::clamp(
        time, 0.0, Level3Config::heat_off_time);
    return Level3Config::source_start_x +
        Level3Config::weld_speed * active_time;
}

struct TransientMovingHeat2D
    : welding::ZeroDirichletScreenedHeat2D {
    double conductivity;
    const welding::FieldSampler2D &previous;
    const welding::MovingGaussianStepSource2D &heat_source;

    TransientMovingHeat2D(
            double alpha, double thermal_conductivity,
            const welding::FieldSampler2D &previous_field,
            const welding::MovingGaussianStepSource2D &source,
            wos::SourceMode mode = wos::SourceMode::Green)
        : ZeroDirichletScreenedHeat2D(alpha),
          conductivity(thermal_conductivity), previous(previous_field),
          heat_source(source) {
        source_mode = mode;
    }

    double source(wos::Point2D point) const {
        const double history =
            screening_parameter_squared() *
            std::max(0.0, previous.sample(point));
        return history + heat_source.volumetric_power_density(point) /
            conductivity;
    }
};

std::size_t probe_index(const wos::Grid &grid, const Probe &probe) {
    const int i = static_cast<int>(std::llround(
        (probe.x - grid.xmin) / (grid.xmax - grid.xmin) * (grid.Nx - 1)));
    const int j = static_cast<int>(std::llround(
        (probe.y - grid.ymin) / (grid.ymax - grid.ymin) * (grid.Ny - 1)));
    const double x = wos::grid_coordinate(grid.xmin, grid.xmax, i, grid.Nx);
    const double y = wos::grid_coordinate(grid.ymin, grid.ymax, j, grid.Ny);
    if (std::abs(x - probe.x) > 1e-12 || std::abs(y - probe.y) > 1e-12) {
        throw std::runtime_error("a fixed Level-3 probe is not on the query grid");
    }
    return welding::flat_index_2d(i, j, grid.Ny);
}

void write_probes(std::ofstream &output, int step, double time,
                  const wos::Grid &grid, const std::vector<double> &field,
                  const std::vector<double> &standard_error) {
    for (const Probe &probe : probes) {
        const std::size_t index = probe_index(grid, probe);
        output << step << ',' << time << ',' << probe.name << ','
               << probe.x << ',' << probe.y << ','
               << probe.source_arrival_time << ',' << field[index] << ','
               << Level3Config::ambient_temperature + field[index] << ','
               << standard_error[index] << '\n';
    }
}

int run_level3(const Level3Config &config) {
#ifdef WELDING_LEVEL3_HAS_OPENMP
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

    const double xmin = -0.5 * Level3Config::length_x;
    const double xmax = 0.5 * Level3Config::length_x;
    const double ymin = -0.5 * Level3Config::length_y;
    const double ymax = 0.5 * Level3Config::length_y;
    const wos::Grid grid{
        Level3Config::nx, Level3Config::ny, 1,
        xmin, xmax, ymin, ymax, 0.0, 0.0,
    };
    const std::size_t point_count =
        static_cast<std::size_t>(grid.Nx) * grid.Ny;
    const int step_count = static_cast<int>(std::llround(
        Level3Config::total_time / Level3Config::dt));
    const double diffusivity = Level3Config::conductivity /
        (Level3Config::density * Level3Config::specific_heat);
    const double alpha = 1.0 /
        std::sqrt(diffusivity * Level3Config::dt);
    const wos::solver::WoS solver(
        {config.walks, Level3Config::epsilon, Level3Config::max_steps});

    if (std::abs(source_x_at(Level3Config::heat_off_time) -
                 Level3Config::source_end_x) > 1e-12) {
        throw std::runtime_error("fixed source trajectory is inconsistent");
    }

    wos::Mesh<2> mesh = welding::make_rectangular_plate(
        xmin, xmax, ymin, ymax);
    wos::FcpwGeometryScene<2> geometry_scene(mesh);
    const std::vector<wos::solver::StartPoint<2>> starts =
        welding::build_start_points_2d(grid, geometry_scene);

    std::vector<double> previous(point_count, 0.0);
    std::vector<double> next(point_count, 0.0);
    std::vector<double> standard_error(point_count, 0.0);

    const std::filesystem::path prefix(config.output_prefix);
    const std::filesystem::path summary_path = prefix.string() + "_summary.csv";
    const std::filesystem::path history_path = prefix.string() + "_history.csv";
    const std::filesystem::path metadata_path = prefix.string() + "_metadata.csv";
    const std::filesystem::path probes_path = prefix.string() + "_probes.csv";
    welding::ensure_parent_directory(summary_path);
    std::ofstream summary_output(summary_path);
    std::ofstream history_output(history_path);
    std::ofstream metadata_output(metadata_path);
    std::ofstream probes_output(probes_path);
    if (!summary_output || !history_output || !metadata_output || !probes_output) {
        throw std::runtime_error("could not open Level-3 output files");
    }
    summary_output << std::setprecision(17);
    history_output << std::setprecision(17);
    metadata_output << std::setprecision(17);
    probes_output << std::setprecision(17);

    metadata_output
        << "key,value,unit\n"
        << "nx," << grid.Nx << ",1\n"
        << "ny," << grid.Ny << ",1\n"
        << "walks," << config.walks << ",1\n"
        << "max_steps," << Level3Config::max_steps << ",1\n"
        << "field_stride," << config.field_stride << ",1\n"
        << "history_interpolation,cubic,1\n"
        << "threads," << effective_threads << ",1\n"
        << "dt," << Level3Config::dt << ",s\n"
        << "total_time," << Level3Config::total_time << ",s\n"
        << "heat_off_time," << Level3Config::heat_off_time << ",s\n"
        << "cooling_duration,"
        << Level3Config::total_time - Level3Config::heat_off_time << ",s\n"
        << "epsilon," << Level3Config::epsilon << ",m\n"
        << "seed," << config.seed << ",1\n"
        << "length_x," << Level3Config::length_x << ",m\n"
        << "length_y," << Level3Config::length_y << ",m\n"
        << "thickness," << Level3Config::thickness << ",m\n"
        << "conductivity," << Level3Config::conductivity << ",W/(m K)\n"
        << "density," << Level3Config::density << ",kg/m3\n"
        << "specific_heat," << Level3Config::specific_heat << ",J/(kg K)\n"
        << "ambient_temperature," << Level3Config::ambient_temperature << ",K\n"
        << "electrical_power," << Level3Config::electrical_power << ",W\n"
        << "efficiency," << Level3Config::efficiency << ",1\n"
        << "absorbed_power,"
        << Level3Config::electrical_power * Level3Config::efficiency << ",W\n"
        << "sigma_x," << Level3Config::sigma_x << ",m\n"
        << "sigma_y," << Level3Config::sigma_y << ",m\n"
        << "source_start_x," << Level3Config::source_start_x << ",m\n"
        << "source_end_x," << Level3Config::source_end_x << ",m\n"
        << "weld_y," << Level3Config::weld_y << ",m\n"
        << "weld_speed," << Level3Config::weld_speed << ",m/s\n"
        << "source_time_quadrature,two_point_gauss_legendre,1\n"
        << "diffusivity," << diffusivity << ",m2/s\n"
        << "alpha," << alpha << ",1/m\n"
        << "source_mode,screened_green,1\n"
        << "nonnegative_history_clamp,true,1\n"
        << "boundary_condition,zero_temperature_rise,1\n";

    summary_output
        << "step,time,phase_code,source_center_x,source_center_y,"
           "source_fraction,absorbed_power,cumulative_input_energy,"
           "center_temperature_rise,center_temperature,center_standard_error,"
           "peak_temperature_rise,peak_temperature,peak_x,peak_y,"
           "mean_standard_error,max_standard_error,"
           "minimum_interior_temperature_rise,negative_interior_points,"
           "thermal_energy_rise,max_steps_hits,step_seconds\n";
    history_output
        << "step,time,x,y,temperature_rise,temperature,standard_error\n";
    probes_output
        << "step,time,probe_id,x,y,source_arrival_time,"
           "temperature_rise,temperature,standard_error\n";

    const std::size_t center_index = welding::flat_index_2d(
        grid.Nx / 2, grid.Ny / 2, grid.Ny);
    welding::write_temperature_field_2d(
        history_output, 0, 0.0, Level3Config::ambient_temperature,
        grid, previous, standard_error);
    write_probes(probes_output, 0, 0.0, grid, previous, standard_error);
    summary_output
        << "0,0,0," << Level3Config::source_start_x << ','
        << Level3Config::weld_y << ",0,0,0,0,"
        << Level3Config::ambient_temperature
        << ",0,0," << Level3Config::ambient_temperature
        << ",0,0,0,0,0,0,0,0,0\n";

    std::printf("Level-3 moving Gaussian heating followed by cooling\n");
    std::printf("  plate / grid:   %.0f x %.0f x %.0f mm / %d x %d\n",
                1e3 * Level3Config::length_x,
                1e3 * Level3Config::length_y,
                1e3 * Level3Config::thickness, grid.Nx, grid.Ny);
    std::printf("  source path:    %.0f to %.0f mm at %.0f mm/s\n",
                1e3 * Level3Config::source_start_x,
                1e3 * Level3Config::source_end_x,
                1e3 * Level3Config::weld_speed);
    std::printf("  moving / cool:  0--%.1f s / %.1f--%.1f s\n",
                Level3Config::heat_off_time,
                Level3Config::heat_off_time,
                Level3Config::total_time);
    std::printf("  dt / steps:     %.3f s / %d\n", Level3Config::dt, step_count);
    std::printf("  absorbed power: %.1f W\n",
                Level3Config::electrical_power * Level3Config::efficiency);
    std::printf("  walks / threads:%d / %d\n\n", config.walks, effective_threads);
    std::printf(" step  time  phase  source x  centre rise  peak rise  energy     seconds\n");

    double cumulative_input_energy = 0.0;
    for (int step = 1; step <= step_count; ++step) {
        const double time_begin = (step - 1) * Level3Config::dt;
        const double time_end = step * Level3Config::dt;
        const welding::MovingGaussianStepSource2D heat_source(
            xmin, xmax, ymin, ymax,
            Level3Config::thickness,
            Level3Config::electrical_power,
            Level3Config::efficiency,
            Level3Config::sigma_x,
            Level3Config::sigma_y,
            Level3Config::weld_y,
            Level3Config::source_start_x,
            Level3Config::weld_speed,
            Level3Config::heat_off_time,
            time_begin, time_end);
        cumulative_input_energy +=
            heat_source.step_average_absorbed_power() * Level3Config::dt;

        welding::FieldSampler2D sampler(
            grid, previous, welding::FieldInterpolation2D::Cubic);
        const TransientMovingHeat2D equation{
            alpha, Level3Config::conductivity, sampler, heat_source,
            wos::SourceMode::Green,
        };
        std::fill(next.begin(), next.end(), 0.0);
        std::fill(standard_error.begin(), standard_error.end(), 0.0);
        long long max_steps_hits = 0;
        const auto step_start = std::chrono::steady_clock::now();

#ifdef WELDING_LEVEL3_HAS_OPENMP
#pragma omp parallel for collapse(2) reduction(+ : max_steps_hits) schedule(static)
#endif
        for (int i = 1; i < grid.Nx - 1; ++i) {
            for (int j = 1; j < grid.Ny - 1; ++j) {
                const double x = wos::grid_coordinate(
                    grid.xmin, grid.xmax, i, grid.Nx);
                const double y = wos::grid_coordinate(
                    grid.ymin, grid.ymax, j, grid.Ny);
                const std::size_t index = welding::flat_index_2d(i, j, grid.Ny);
                welding::PathRandomStreams random =
                    welding::make_path_random_streams(welding::point_seed(
                        config.seed,
                        wos::splitmix64(static_cast<std::uint64_t>(step)),
                        index));
                const wos::solver::Result result = solver.solve(
                    geometry_scene, wos::Point2D{x, y}, starts[index], equation,
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
                grid, next, standard_error, Level3Config::density,
                Level3Config::specific_heat, Level3Config::thickness);
        const double peak_x = wos::grid_coordinate(
            grid.xmin, grid.xmax, field_summary.peak_i, grid.Nx);
        const double peak_y = wos::grid_coordinate(
            grid.ymin, grid.ymax, field_summary.peak_j, grid.Ny);
        const int phase_code = time_end <= Level3Config::heat_off_time ? 1 : 2;
        const double reported_source_x = source_x_at(time_end);

        summary_output
            << step << ',' << time_end << ',' << phase_code << ','
            << reported_source_x << ',' << Level3Config::weld_y << ','
            << heat_source.active_fraction() << ','
            << heat_source.step_average_absorbed_power() << ','
            << cumulative_input_energy << ','
            << next[center_index] << ','
            << Level3Config::ambient_temperature + next[center_index] << ','
            << standard_error[center_index] << ','
            << field_summary.peak_rise << ','
            << Level3Config::ambient_temperature + field_summary.peak_rise << ','
            << peak_x << ',' << peak_y << ','
            << field_summary.mean_standard_error << ','
            << field_summary.max_standard_error << ','
            << field_summary.minimum_interior_rise << ','
            << field_summary.negative_interior_points << ','
            << field_summary.thermal_energy_rise << ','
            << max_steps_hits << ',' << step_seconds << '\n';
        write_probes(probes_output, step, time_end, grid, next, standard_error);

        if (step % config.field_stride == 0 || step == step_count ||
            std::abs(time_end - Level3Config::heat_off_time) < 0.5 * Level3Config::dt) {
            welding::write_temperature_field_2d(
                history_output, step, time_end,
                Level3Config::ambient_temperature,
                grid, next, standard_error);
        }

        const int report_stride = std::max(1, step_count / 20);
        if (step == 1 || step == step_count || step % report_stride == 0 ||
            std::abs(time_end - Level3Config::heat_off_time) < 0.5 * Level3Config::dt) {
            std::printf("%5d %5.1f %6s %9.1f %12.4f %10.4f %9.3f %9.3f\n",
                        step, time_end, phase_code == 1 ? "move" : "cool",
                        1e3 * reported_source_x, next[center_index],
                        field_summary.peak_rise,
                        field_summary.thermal_energy_rise, step_seconds);
        }
        previous.swap(next);
    }

    const double expected_input_energy =
        Level3Config::electrical_power * Level3Config::efficiency *
        Level3Config::heat_off_time;
    std::printf("\nFinished Level 3.\n");
    std::printf("  cumulative source energy: %.12g J\n", cumulative_input_energy);
    std::printf("  expected source energy:   %.12g J\n", expected_input_energy);
    std::printf("  source-energy error:      %.12g J\n",
                cumulative_input_energy - expected_input_energy);
    std::printf("  final centre rise:        %.12g K\n", previous[center_index]);
    std::printf("  summary: %s\n", summary_path.string().c_str());
    std::printf("  probes:  %s\n", probes_path.string().c_str());
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
                         "wos_welding_level3 supports one MPI rank; got %d\n",
                         size);
        }
        status = 1;
    } else {
        try {
            status = run_level3(parse_options(argc, argv));
        } catch (const std::exception &error) {
            std::fprintf(stderr, "wos_welding_level3: %s\n", error.what());
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
