#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <mpi.h>

#include "welding/field_sampler.hpp"
#include "wos/boundary/condition.hpp"
#include "wos/bvh.hpp"
#include "wos/geometry/sphere.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"
#include "wos/sampling/screened_green.hpp"
#include "wos/solver/wos.hpp"
#include "wos/source_mode.hpp"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

#ifndef WELDING_RESULTS_DIR
#define WELDING_RESULTS_DIR "apps/welding/results"
#endif

struct Options {
    int grid_size = 17;
    int steps = 5;
    int walks = 2'000;
    int max_steps = 1'000;
    double dt = 0.01;
    double diffusivity = 1.0;
    double epsilon = 1e-3;
    std::uint64_t seed = 12345;
    std::string output = WELDING_RESULTS_DIR "/data/level0_cooling.csv";
};

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Level-0 transient cooling verification on the unit square.\n"
        "The initial temperature rise is sin(pi*x) sin(pi*y), and all\n"
        "boundary temperature rises are zero. This executable currently\n"
        "supports exactly one MPI rank.\n"
        "\n"
        "Options:\n"
        "  --grid N          odd grid size in each direction (default: 17)\n"
        "  --steps N         number of implicit Euler steps (default: 5)\n"
        "  --dt X            time-step size (default: 0.01)\n"
        "  --diffusivity X   thermal diffusivity (default: 1.0)\n"
        "  --walks N         WoS paths per interior point (default: 2000)\n"
        "  --epsilon X       boundary stopping distance (default: 0.001)\n"
        "  --max-steps N     maximum sphere steps per path (default: 1000)\n"
        "  --seed N          reproducible unsigned seed (default: 12345)\n"
        "  --output FILE     time-summary CSV path\n"
        "  -h, --help        show this message\n",
        program);
}

int parse_positive_int(const char *text, const char *name) {
    int value = 0;
    const char *end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value <= 0) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

double parse_positive_double(const char *text, const char *name) {
    if (*text == '\0') {
        throw std::invalid_argument(std::string("invalid ") + name + ": empty value");
    }
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

std::uint64_t parse_uint64(const char *text) {
    std::uint64_t value = 0;
    const char *end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument(std::string("invalid seed: ") + text);
    }
    return value;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value_after = [&](const char *name) -> const char * {
            if (++i >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[i];
        };

        if (argument == "--grid") {
            options.grid_size = parse_positive_int(value_after("--grid"), "grid size");
        } else if (argument == "--steps") {
            options.steps = parse_positive_int(value_after("--steps"), "step count");
        } else if (argument == "--dt") {
            options.dt = parse_positive_double(value_after("--dt"), "time step");
        } else if (argument == "--diffusivity") {
            options.diffusivity = parse_positive_double(
                value_after("--diffusivity"), "diffusivity");
        } else if (argument == "--walks") {
            options.walks = parse_positive_int(value_after("--walks"), "walk count");
        } else if (argument == "--epsilon") {
            options.epsilon = parse_positive_double(
                value_after("--epsilon"), "epsilon");
        } else if (argument == "--max-steps") {
            options.max_steps = parse_positive_int(
                value_after("--max-steps"), "maximum step count");
        } else if (argument == "--seed") {
            options.seed = parse_uint64(value_after("--seed"));
        } else if (argument == "--output") {
            options.output = value_after("--output");
            if (options.output.empty()) {
                throw std::invalid_argument("--output requires a non-empty path");
            }
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.grid_size < 3 || options.grid_size % 2 == 0) {
        throw std::invalid_argument("--grid must be an odd integer of at least 3");
    }
    if (!(options.epsilon < 0.5)) {
        throw std::invalid_argument("--epsilon must be smaller than 0.5");
    }
    return options;
}

wos::Mesh<2> make_unit_square() {
    wos::Mesh<2> mesh;
    mesh.verts = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 1.0},
    };
    mesh.prims = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
    };
    mesh.boundary_ids = {0, 0, 0, 0};
    return mesh;
}

std::size_t flat_index(int i, int j, int ny) {
    return static_cast<std::size_t>(i) * ny + j;
}

double eigenfunction(double x, double y) {
    return std::sin(pi * x) * std::sin(pi * y);
}

struct TransientCoolingEquation2D {
    static constexpr bool has_source = true;
    static constexpr bool has_screening = true;
    static constexpr bool has_green_source = true;

    double alpha;
    const welding::FieldSampler2D &previous;
    wos::SourceMode source_mode = wos::SourceMode::Green;

    bool source_may_intersect([[maybe_unused]] wos::Sphere2D sphere) const {
        return true;
    }

    double source(wos::Point2D point) const {
        return alpha * alpha * previous.sample(point);
    }

    wos::BoundaryCondition boundary(
            [[maybe_unused]] wos::Point2D point,
            [[maybe_unused]] int boundary_id) const {
        return wos::BoundaryCondition::dirichlet(0.0);
    }

    double green(wos::Sphere2D sphere, wos::Point2D x, wos::Point2D y) const {
        return wos::screened_green::green_2d(
            alpha, sphere.radius, wos::dist(x, y));
    }

    double screening_factor(double radius) const {
        return wos::screened_green::weight_2d(alpha, radius);
    }

    double green_mass(double radius) const {
        return wos::screened_green::mass_2d(alpha, radius);
    }

    wos::Point2D sample_green(wos::Sphere2D sphere, wos::PRNG &rng) const {
        const double radius = wos::screened_green::sample_radius_2d(
            alpha, sphere.radius, rng);
        const double angle = 2.0 * pi * rng.unit();
        return wos::Point2D{
            sphere.centre.x + radius * std::cos(angle),
            sphere.centre.y + radius * std::sin(angle),
        };
    }
};

struct ErrorSummary {
    double rmse;
    double max_absolute;
};

ErrorSummary field_error(const wos::Grid &grid,
                         const std::vector<double> &field,
                         double amplitude) {
    double squared_sum = 0.0;
    double max_absolute = 0.0;
    std::size_t count = 0;
    for (int i = 1; i < grid.Nx - 1; ++i) {
        const double x = wos::grid_coordinate(grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 1; j < grid.Ny - 1; ++j) {
            const double y = wos::grid_coordinate(grid.ymin, grid.ymax, j, grid.Ny);
            const double exact = amplitude * eigenfunction(x, y);
            const double error = field[flat_index(i, j, grid.Ny)] - exact;
            squared_sum += error * error;
            max_absolute = std::max(max_absolute, std::abs(error));
            ++count;
        }
    }
    return ErrorSummary{
        std::sqrt(squared_sum / static_cast<double>(count)),
        max_absolute,
    };
}

int run_level0(const Options &options) {
    const wos::Grid grid{
        options.grid_size, options.grid_size, 1,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 0.0,
    };
    const std::size_t point_count =
        static_cast<std::size_t>(grid.Nx) * grid.Ny;

    wos::Mesh<2> mesh = make_unit_square();
    std::unique_ptr<wos::BVH<2>> bvh = wos::build_bvh(mesh);

    std::vector<wos::solver::StartPoint<2>> starts(point_count);
    std::vector<double> previous(point_count, 0.0);
    std::vector<double> next(point_count, 0.0);
    std::vector<double> standard_error(point_count, 0.0);

    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = flat_index(i, j, grid.Ny);
            const wos::Point2D point{x, y};
            starts[index] = wos::solver::find_start_point(*bvh, point);
            if (i != 0 && i != grid.Nx - 1 && j != 0 && j != grid.Ny - 1) {
                previous[index] = eigenfunction(x, y);
            }
        }
    }

    const double alpha = 1.0 / std::sqrt(options.diffusivity * options.dt);
    const double decay_per_step = 1.0 /
        (1.0 + 2.0 * pi * pi * options.diffusivity * options.dt);
    const wos::solver::WoS solver(
        {options.walks, options.epsilon, options.max_steps});

    const std::filesystem::path output_path(options.output);
    if (!output_path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(output_path.parent_path(), error);
        if (error) {
            throw std::runtime_error(
                "could not create output directory: " + error.message());
        }
    }
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("could not open output file: " + options.output);
    }
    output << "step,time,center_mean,center_standard_error,"
              "center_exact_discrete,center_exact_continuous,"
              "center_absolute_error,field_rmse,field_max_absolute_error\n";

    std::filesystem::path history_path = output_path;
    history_path.replace_filename(
        output_path.stem().string() + "_history" + output_path.extension().string());
    std::ofstream history_output(history_path);
    if (!history_output) {
        throw std::runtime_error(
            "could not open history output file: " + history_path.string());
    }
    history_output << "step,time,x,y,numerical,standard_error,"
                      "exact_discrete,error,absolute_error\n";
    auto write_history_step = [&](int step, double time,
                                  const std::vector<double> &field,
                                  const std::vector<double> &field_standard_error,
                                  double exact_amplitude) {
        for (int i = 0; i < grid.Nx; ++i) {
            const double x = wos::grid_coordinate(
                grid.xmin, grid.xmax, i, grid.Nx);
            for (int j = 0; j < grid.Ny; ++j) {
                const double y = wos::grid_coordinate(
                    grid.ymin, grid.ymax, j, grid.Ny);
                const std::size_t index = flat_index(i, j, grid.Ny);
                const double exact = exact_amplitude * eigenfunction(x, y);
                const double error = field[index] - exact;
                history_output << step << ','
                               << time << ','
                               << x << ','
                               << y << ','
                               << field[index] << ','
                               << field_standard_error[index] << ','
                               << exact << ','
                               << error << ','
                               << std::abs(error) << '\n';
            }
        }
    };

    const int center = grid.Nx / 2;
    const std::size_t center_index = flat_index(center, center, grid.Ny);
    output << "0,0," << previous[center_index]
           << ",0,1,1,0,0,0\n";
    write_history_step(0, 0.0, previous, standard_error, 1.0);

    std::printf("Level-0 transient cooling verification\n");
    std::printf("  grid:          %d x %d\n", grid.Nx, grid.Ny);
    std::printf("  time step:     %.12g\n", options.dt);
    std::printf("  diffusivity:   %.12g\n", options.diffusivity);
    std::printf("  alpha:         %.12g\n", alpha);
    std::printf("  walks/point:   %d\n", options.walks);
    std::printf("  output:        %s\n\n", options.output.c_str());
    std::printf(" step       time       center mean    discrete exact    abs error      std error       field RMSE\n");
    std::printf("%5d %11.5g %17.9g %17.9g %13.5g %13.5g %16.5g\n",
                0, 0.0, previous[center_index], 1.0, 0.0, 0.0, 0.0);

    for (int step = 1; step <= options.steps; ++step) {
        welding::FieldSampler2D sampler(grid, previous);
        const TransientCoolingEquation2D equation{alpha, sampler};
        std::fill(next.begin(), next.end(), 0.0);
        std::fill(standard_error.begin(), standard_error.end(), 0.0);

        for (int i = 1; i < grid.Nx - 1; ++i) {
            const double x = wos::grid_coordinate(
                grid.xmin, grid.xmax, i, grid.Nx);
            for (int j = 1; j < grid.Ny - 1; ++j) {
                const double y = wos::grid_coordinate(
                    grid.ymin, grid.ymax, j, grid.Ny);
                const std::size_t index = flat_index(i, j, grid.Ny);
                const wos::Point2D point{x, y};

                const std::uint64_t point_seed = wos::splitmix64(
                    options.seed ^
                    wos::splitmix64(static_cast<std::uint64_t>(step)) ^
                    wos::splitmix64(static_cast<std::uint64_t>(index)));
                wos::PRNG walk_rng(wos::splitmix64(
                    point_seed ^ 0x13198A2E03707344ULL));
                wos::PRNG source_rng(wos::splitmix64(
                    point_seed ^ 0xA4093822299F31D0ULL));

                const wos::solver::Result result = solver.solve(
                    *bvh, point, starts[index], equation,
                    walk_rng, source_rng);
                next[index] = result.mean;
                standard_error[index] = result.standard_error;
            }
        }

        const double time = step * options.dt;
        const double discrete_amplitude = std::pow(decay_per_step, step);
        const double continuous_amplitude = std::exp(
            -2.0 * pi * pi * options.diffusivity * time);
        const ErrorSummary errors = field_error(
            grid, next, discrete_amplitude);
        const double center_error = std::abs(
            next[center_index] - discrete_amplitude);

        output << step << ','
               << time << ','
               << next[center_index] << ','
               << standard_error[center_index] << ','
               << discrete_amplitude << ','
               << continuous_amplitude << ','
               << center_error << ','
               << errors.rmse << ','
               << errors.max_absolute << '\n';
        write_history_step(
            step, time, next, standard_error, discrete_amplitude);

        std::printf("%5d %11.5g %17.9g %17.9g %13.5g %13.5g %16.5g\n",
                    step, time, next[center_index], discrete_amplitude,
                    center_error, standard_error[center_index], errors.rmse);
        previous.swap(next);
    }

    std::filesystem::path field_path = output_path;
    field_path.replace_filename(
        output_path.stem().string() + "_field" + output_path.extension().string());
    std::ofstream field_output(field_path);
    if (!field_output) {
        throw std::runtime_error(
            "could not open field output file: " + field_path.string());
    }
    field_output << "x,y,numerical,standard_error,exact_discrete,error,absolute_error\n";
    const double final_amplitude = std::pow(decay_per_step, options.steps);
    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index = flat_index(i, j, grid.Ny);
            const double exact = final_amplitude * eigenfunction(x, y);
            const double error = previous[index] - exact;
            field_output << x << ','
                         << y << ','
                         << previous[index] << ','
                         << standard_error[index] << ','
                         << exact << ','
                         << error << ','
                         << std::abs(error) << '\n';
        }
    }

    std::printf("\nFinished.\n");
    std::printf("  time summary: %s\n", options.output.c_str());
    std::printf("  full history: %s\n", history_path.string().c_str());
    std::printf("  final field:  %s\n", field_path.string().c_str());
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
                         "wos_welding_level0 currently supports one MPI rank; got %d\n",
                         size);
        }
        status = 1;
    } else {
        try {
            status = run_level0(parse_options(argc, argv));
        } catch (const std::exception &error) {
            std::fprintf(stderr, "wos_welding_level0: %s\n", error.what());
            status = 1;
        }
    }

    MPI_Finalize();
    return status;
}
