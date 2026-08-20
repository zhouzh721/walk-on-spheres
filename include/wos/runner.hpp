#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mpi.h>
#include "wos/bvh.hpp"
#include "wos/field_3D.hpp"
#include "wos/grid.hpp"
#include "wos/hash.hpp"
#include "wos/hdf5_io.hpp"
#include "wos/inside.hpp"
#include "wos/mesh.hpp"
#include "wos/prng.hpp"
#include "wos/solver/wos.hpp"
#include "wos/source_mode.hpp"

namespace wos {

// MPI-parallel sweep of WoS over a grid covering the mesh's bounding box
template<int N, typename Eq>
int run(int rank, int size, const char *mesh_filename, const char *output_filename,
        int Nx, int Ny, int Nz, int N_walks, double epsilon, int max_steps,
        int max_ray_attempts, const Eq &eq, std::uint64_t global_seed,
        bool write_source_metadata = false, double alpha = 0.0,
        SourceMode source_mode = SourceMode::Uniform) {
    Mesh<N> domain;
    int mesh_ok = 1;
    if (rank == 0) {
        std::printf("Reading mesh from disk (%s)...\n", mesh_filename);
        try {
            domain = load_mesh<N>(mesh_filename);
            std::printf("Finished reading. Broadcasting mesh...\n");
        } catch (const std::exception &error) {
            std::fprintf(stderr, "%s\n", error.what());
            mesh_ok = 0;
        }
    }
    MPI_Bcast(&mesh_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!mesh_ok) {
        return 1;
    }
    bcast_mesh(domain, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Finished broadcasting.\n");

    std::unique_ptr<BVH<N>> bvh;
    int bvh_ok = 1;
    double bvh_build_seconds = 0.0;
    if (rank == 0) {
        std::printf("Building BVH...\n");
        const double build_start = MPI_Wtime();
        try {
            bvh = build_bvh(domain);
            bvh_build_seconds = MPI_Wtime() - build_start;
        } catch (const std::exception &error) {
            std::fprintf(stderr, "BVH build failed: %s\n", error.what());
            bvh_ok = 0;
        }
    }
    MPI_Bcast(&bvh_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!bvh_ok) {
        return 1;
    }
    if (rank != 0) bvh = std::make_unique<BVH<N>>();

    if (rank == 0) {
        if (size > 1) {
            std::printf("Finished BVH build in %.6f s. Broadcasting BVH...\n",
                        bvh_build_seconds);
        } else {
            std::printf("Finished BVH build in %.6f s.\n", bvh_build_seconds);
        }
    }
    const double bcast_start = MPI_Wtime();
    bcast_bvh(*bvh, domain, 0, MPI_COMM_WORLD);
    const double bcast_seconds = MPI_Wtime() - bcast_start;
    if (rank == 0 && size > 1) {
        std::printf("Finished BVH broadcast in %.6f s.\n", bcast_seconds);
    }

    // setup domain discretisation
    if constexpr (N == 2) Nz = 1;
    Grid grid{Nx, Ny, Nz, 0,0, 0,0, 0,0};
    mesh_bbox(domain, &grid.xmin, &grid.xmax, &grid.ymin, &grid.ymax, &grid.zmin, &grid.zmax);
    double mesh_scale = std::max({1.0, grid.xmax - grid.xmin,
                                  grid.ymax - grid.ymin, grid.zmax - grid.zmin});
    double boundary_tolerance = 1e-10 * mesh_scale;

    // 3D Cartesian decomposition: split ranks across x, y, z axes
    const long long grid_point_count =
        static_cast<long long>(grid.Nx) * grid.Ny * grid.Nz;
    if (size > grid_point_count) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Cannot assign %d MPI ranks to only %lld grid points; "
                         "use fewer ranks or a larger grid\n",
                         size, grid_point_count);
        }
        return 1;
    }

    int dims[3] = { 0, 0, 0 };
    if (grid.Nx == 1) dims[0] = 1;
    if (grid.Ny == 1) dims[1] = 1;
    if constexpr (N == 2) {
        dims[2] = 1;
    } else if (grid.Nz == 1) {
        dims[2] = 1;
    }
    MPI_Dims_create(size, 3, dims);
    if (dims[0] > grid.Nx || dims[1] > grid.Ny || dims[2] > grid.Nz) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI decomposition %d x %d x %d exceeds grid %d x %d x %d; "
                         "use fewer ranks or a larger grid\n",
                         dims[0], dims[1], dims[2], grid.Nx, grid.Ny, grid.Nz);
        }
        return 1;
    }

    int cx = rank / (dims[1] * dims[2]);
    int cy = (rank / dims[2]) % dims[1];
    int cz = rank % dims[2];

    int x_start = (grid.Nx * cx) / dims[0];
    int x_end   = (grid.Nx * (cx + 1)) / dims[0];
    int y_start = (grid.Ny * cy) / dims[1];
    int y_end   = (grid.Ny * (cy + 1)) / dims[1];
    int z_start = (grid.Nz * cz) / dims[2];
    int z_end   = (grid.Nz * (cz + 1)) / dims[2];
    int block_Nx = x_end - x_start;
    int block_Ny = y_end - y_start;
    int block_Nz = z_end - z_start;

    if (rank == 0) {
        std::printf("Parallelisation: %d x %d x %d ranks across (Nx=%d, Ny=%d, Nz=%d)\n",
                    dims[0], dims[1], dims[2], grid.Nx, grid.Ny, grid.Nz);
    }

    Field3D mean(block_Nx, block_Ny, block_Nz);
    Field3D variance(block_Nx, block_Ny, block_Nz);
    Field3D standard_error(block_Nx, block_Ny, block_Nz);
    Field3D mean_steps(block_Nx, block_Ny, block_Nz);
    LocationField3D location(block_Nx, block_Ny, block_Nz);

    long long local_inside_count = 0;
    long long local_max_steps_hits = 0;
    long long local_ambiguous_ray_retries = 0;
    long long local_indeterminate_points = 0;
    double local_sums[3] = {0.0, 0.0, 0.0};

    if (rank == 0) std::printf("Beginning walk-on-spheres...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double wos_start = MPI_Wtime();
    const solver::WoS wos_solver(
        solver::Settings{N_walks, epsilon, max_steps});

    for (int i = 0; i < block_Nx; i++) {
        double x = grid_coordinate(grid.xmin, grid.xmax, x_start + i, grid.Nx);
        for (int j = 0; j < block_Ny; j++) {
            double y = grid_coordinate(grid.ymin, grid.ymax, y_start + j, grid.Ny);
            for (int k = 0; k < block_Nz; k++) {
                double z = grid_coordinate(grid.zmin, grid.zmax, z_start + k, grid.Nz);
                std::uint64_t global_i = static_cast<std::uint64_t>(x_start + i);
                std::uint64_t global_j = static_cast<std::uint64_t>(y_start + j);
                std::uint64_t global_k = static_cast<std::uint64_t>(z_start + k);
                std::uint64_t linear_index = (global_i * static_cast<std::uint64_t>(grid.Ny) + global_j)
                                           * static_cast<std::uint64_t>(grid.Nz) + global_k;
                std::uint64_t point_seed = splitmix64(global_seed ^ splitmix64(linear_index));

                Point<N> p0;
                if constexpr (N == 2) {
                    p0 = Point2D{x, y};
                } else {
                    p0 = Point3D{x, y, z};
                }

                auto start = solver::find_start_point(*bvh, p0);
                PointLocation point_location;
                if constexpr (N == 2) {
                    point_location = classify_point(domain, p0, start.radius, boundary_tolerance);
                } else {
                    constexpr std::uint64_t INSIDE_STREAM = 0x243F6A8885A308D3ULL;
                    PRNG inside_rng(splitmix64(point_seed ^ INSIDE_STREAM));
                    const PointClassification classification =
                        classify_point(*bvh, p0, start.radius, boundary_tolerance,
                                       max_ray_attempts, inside_rng);
                    point_location = classification.location;
                    local_ambiguous_ray_retries +=
                        classification.ambiguous_ray_retries;
                    if (point_location == PointLocation::Indeterminate) {
                        ++local_indeterminate_points;
                    }
                }

                solver::Result point_result{NAN, NAN, NAN, NAN, 0};
                if (point_location == PointLocation::Boundary) {
                    if constexpr (N == 2) {
                        point_result = solver::Result{
                            dirichlet_value(eq.boundary(
                                start.nearest, start.boundary_id)),
                            0.0, 0.0, 0.0, 0
                        };
                    } else {
                        point_result = solver::Result{
                            dirichlet_value(eq.boundary(start.nearest)),
                            0.0, 0.0, 0.0, 0
                        };
                    }
                } else if (point_location == PointLocation::Inside) {
                    constexpr std::uint64_t WOS_STREAM = 0x13198A2E03707344ULL;
                    PRNG walk_rng(splitmix64(point_seed ^ WOS_STREAM));
                    if constexpr (N == 2) {
                        constexpr std::uint64_t SOURCE_STREAM = 0xA4093822299F31D0ULL;
                        PRNG source_rng(splitmix64(point_seed ^ SOURCE_STREAM));
                        point_result = wos_solver.solve(
                            *bvh, p0, start, eq, walk_rng, source_rng);
                    } else {
                        // Preserve the existing 3D behaviour: source sampling
                        // and WoS steps continue to share one random stream.
                        point_result = wos_solver.solve(
                            *bvh, p0, start, eq, walk_rng, walk_rng);
                    }

                    ++local_inside_count;
                    local_max_steps_hits += point_result.max_steps_hits;
                    local_sums[0] += point_result.variance;
                    local_sums[1] += point_result.standard_error;
                    local_sums[2] += point_result.mean_steps;
                }

                mean(i, j, k) = point_result.mean;
                variance(i, j, k) = point_result.variance;
                standard_error(i, j, k) = point_result.standard_error;
                mean_steps(i, j, k) = point_result.mean_steps;
                location(i, j, k) = static_cast<std::uint8_t>(point_location);
            }
        }
    }

    const double local_wos_seconds = MPI_Wtime() - wos_start;
    double global_wos_seconds = 0.0;
    MPI_Reduce(&local_wos_seconds, &global_wos_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    long long global_inside_count = 0;
    long long global_max_steps_hits = 0;
    long long global_ambiguous_ray_retries = 0;
    long long global_indeterminate_points = 0;
    double global_sums[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(&local_inside_count, &global_inside_count, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_steps_hits, &global_max_steps_hits, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sums, global_sums, 3, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if constexpr (N == 3) {
        MPI_Reduce(&local_ambiguous_ray_retries,
                   &global_ambiguous_ray_retries, 1,
                   MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_indeterminate_points,
                   &global_indeterminate_points, 1,
                   MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && global_inside_count > 0) {
        double count = static_cast<double>(global_inside_count);
        std::printf("\nWoS inside-grid averages:\n");
        std::printf("  number of inside points:       %lld\n", global_inside_count);
        std::printf("  average variance:              %.12g\n", global_sums[0] / count);
        std::printf("  average standard error:        %.12g\n", global_sums[1] / count);
        std::printf("  average steps per walk:        %.12g\n", global_sums[2] / count);
        std::printf("  paths reaching max steps:      %lld\n", global_max_steps_hits);
        std::printf("\n");
    }
    if constexpr (N == 3) {
        if (rank == 0) {
            std::printf("3D ray-classification diagnostics:\n");
            std::printf("  ambiguous ray retries:         %lld\n",
                        global_ambiguous_ray_retries);
            std::printf("  indeterminate grid points:     %lld\n\n",
                        global_indeterminate_points);
        }
    }

    if (rank == 0) {
        std::printf("Finished walk-on-spheres in %.6f s. Writing results...\n",
                    global_wos_seconds);
    }

    const bool write_ok =
        wos_write_hdf5(output_filename, grid, x_start, y_start, z_start,
                       mean, variance, standard_error, mean_steps, location,
                       N_walks, epsilon, max_steps, global_max_steps_hits,
                       N == 3, max_ray_attempts,
                       global_ambiguous_ray_retries,
                       global_indeterminate_points,
                       global_seed, write_source_metadata,
                       alpha, source_mode);
    if (!write_ok) {
        if (rank == 0) {
            std::fprintf(stderr, "Failed writing results to %s.\n", output_filename);
        }
        return 1;
    }
    if (rank == 0) std::printf("Finished writing to %s.\n", output_filename);

    return 0;
}

}
