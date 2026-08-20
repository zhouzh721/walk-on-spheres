#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include "welding/cli.hpp"
#include "welding/file_output.hpp"
#include "welding/field_sampler.hpp"
#include "welding/grid_utils.hpp"
#include "welding/random_streams.hpp"
#include "welding/rectangular_plate.hpp"
#include "welding/screened_heat.hpp"
#include "wos/geometry/fcpw_scene.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/mesh.hpp"
#include "wos/solver/wos.hpp"

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

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value_after = [&](const char *name) {
            return welding::option_value(i, argc, argv, name);
        };

        if (argument == "--grid") {
            options.grid_size = welding::parse_positive_int(
                value_after("--grid"), "grid size");
        } else if (argument == "--steps") {
            options.steps = welding::parse_positive_int(
                value_after("--steps"), "step count");
        } else if (argument == "--dt") {
            options.dt = welding::parse_positive_double(
                value_after("--dt"), "time step");
        } else if (argument == "--diffusivity") {
            options.diffusivity = welding::parse_positive_double(
                value_after("--diffusivity"), "diffusivity");
        } else if (argument == "--walks") {
            options.walks = welding::parse_positive_int(
                value_after("--walks"), "walk count");
        } else if (argument == "--epsilon") {
            options.epsilon = welding::parse_positive_double(
                value_after("--epsilon"), "epsilon");
        } else if (argument == "--max-steps") {
            options.max_steps = welding::parse_positive_int(
                value_after("--max-steps"), "maximum step count");
        } else if (argument == "--seed") {
            options.seed = welding::parse_uint64(value_after("--seed"));
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

double eigenfunction(double x, double y) {
    return std::sin(pi * x) * std::sin(pi * y);
}

struct TransientCoolingEquation2D
    : welding::ZeroDirichletScreenedHeat2D {
    const welding::FieldSampler2D &previous;

    TransientCoolingEquation2D(
            double alpha, const welding::FieldSampler2D &previous_field)
        : ZeroDirichletScreenedHeat2D(alpha), previous(previous_field) {}

    double source(wos::Point2D point) const {
        return screening_parameter_squared() * previous.sample(point);
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
            const double error =
                field[welding::flat_index_2d(i, j, grid.Ny)] - exact;
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

    wos::Mesh<2> mesh = welding::make_rectangular_plate(
        grid.xmin, grid.xmax, grid.ymin, grid.ymax);
    wos::FcpwGeometryScene<2> geometry_scene(mesh);

    const std::vector<wos::solver::StartPoint<2>> starts =
        welding::build_start_points_2d(grid, geometry_scene);
    std::vector<double> previous(point_count, 0.0);
    std::vector<double> next(point_count, 0.0);
    std::vector<double> standard_error(point_count, 0.0);

    for (int i = 0; i < grid.Nx; ++i) {
        const double x = wos::grid_coordinate(grid.xmin, grid.xmax, i, grid.Nx);
        for (int j = 0; j < grid.Ny; ++j) {
            const double y = wos::grid_coordinate(grid.ymin, grid.ymax, j, grid.Ny);
            const std::size_t index =
                welding::flat_index_2d(i, j, grid.Ny);
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
    welding::ensure_parent_directory(output_path);
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
                    const std::size_t index =
                        welding::flat_index_2d(i, j, grid.Ny);
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
    const std::size_t center_index =
        welding::flat_index_2d(center, center, grid.Ny);
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
                const std::size_t index =
                    welding::flat_index_2d(i, j, grid.Ny);
                const wos::Point2D point{x, y};

                welding::PathRandomStreams random =
                    welding::make_path_random_streams(welding::point_seed(
                        options.seed,
                        wos::splitmix64(static_cast<std::uint64_t>(step)),
                        index));

                const wos::solver::Result result = solver.solve(
                    geometry_scene, point, starts[index], equation,
                    random.walk, random.source);
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
            const std::size_t index =
                welding::flat_index_2d(i, j, grid.Ny);
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
