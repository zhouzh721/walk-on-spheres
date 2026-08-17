// Walk-on-spheres solver launcher
// CLI: wos [options] [equation] [Nx Ny] [Nz] [mesh.obj]
#include <charconv>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <exception>
#include <system_error>
#include <vector>
#include <mpi.h>
#include "equations/equations.hpp"
#include "wos/mesh.hpp"

using namespace wos;

static void print_usage(const char *program) {
    std::printf(
        "Usage: %s [options] [equation] [Nx Ny] [Nz] [mesh.obj]\n"
        "\n"
        "Options:\n"
        "  -o, --output FILE  HDF5 output path (default: wos.h5)\n"
        "      --seed UINT64  fixed random seed (default: generated from current time)\n"
        "      --walks N      WoS paths per inside grid point (default: 10000)\n"
        "      --epsilon X    boundary stopping distance (default: 0.01)\n"
        "      --max-steps N  maximum steps per WoS path (default: 1000)\n"
        "      --max-ray-attempts N\n"
        "                     maximum 3D inside-test ray attempts (default: 12)\n"
        "      --alpha X      screened Poisson coefficient (default: 5;\n"
        "                     screened_poisson only)\n"
        "      --source-mode MODE\n"
        "                     screened source sampling: uniform or green\n"
        "                     (screened_poisson only)\n"
        "      --force        allow an existing output file to be overwritten\n"
        "  -h, --help         show this help message\n",
        program
    );
}

static bool parse_int(const char *text, int minimum, int *value) {
    int parsed = 0;
    const char *end = text + std::strlen(text);
    const std::from_chars_result result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed < minimum) {
        return false;
    }
    *value = parsed;
    return true;
}

static bool parse_uint64(const char *text, std::uint64_t *value) {
    std::uint64_t parsed = 0;
    const char *end = text + std::strlen(text);
    const std::from_chars_result result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    *value = parsed;
    return true;
}

static bool parse_positive_double(const char *text, double *value) {
    if (*text == '\0' ||
        std::isspace(static_cast<unsigned char>(*text))) {
        return false;
    }

    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed) || parsed <= 0.0) {
        return false;
    }
    *value = parsed;
    return true;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // defaults
    const char *eq_name = "laplace";
    const char *mesh_filename = "meshes/annulus.obj";
    const char *output_filename = "wos.h5";
    int Nx = 32, Ny = 32, Nz = 32;
    int N_walks = 10'000;
    double epsilon = 1e-2;
    int max_steps = 1'000;
    int max_ray_attempts = 12;
    double alpha = 5.0;
    SourceMode source_mode = SourceMode::Uniform;
    bool force_output = false;
    std::uint64_t seed = 0;
    bool seed_provided = false;
    bool alpha_provided = false;
    bool source_mode_provided = false;

    // Options may appear before or after the positional arguments.
    int cli_status = 0; // 0 = continue, 1 = error, 2 = help shown
    if (rank == 0) {
        std::vector<const char *> positional;
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                cli_status = 2;
                break;
            } else if (std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires a file path\n", argv[i-1]);
                    cli_status = 1;
                    break;
                }
                output_filename = argv[i];
            } else if (std::strcmp(argv[i], "--seed") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires an unsigned 64-bit integer\n", argv[i-1]);
                    cli_status = 1;
                    break;
                }
                const char *seed_text = argv[i];
                if (!parse_uint64(seed_text, &seed)) {
                    std::fprintf(stderr, "Invalid --seed value: %s\n", seed_text);
                    cli_status = 1;
                    break;
                }
                seed_provided = true;
            } else if (std::strcmp(argv[i], "--walks") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires a positive integer\n", argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (!parse_int(argv[i], 1, &N_walks)) {
                    std::fprintf(stderr, "Invalid --walks value: %s\n", argv[i]);
                    cli_status = 1;
                    break;
                }
            } else if (std::strcmp(argv[i], "--epsilon") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires a positive finite number\n",
                                 argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (!parse_positive_double(argv[i], &epsilon)) {
                    std::fprintf(stderr, "Invalid --epsilon value: %s\n", argv[i]);
                    cli_status = 1;
                    break;
                }
            } else if (std::strcmp(argv[i], "--max-steps") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires a positive integer\n", argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (!parse_int(argv[i], 1, &max_steps)) {
                    std::fprintf(stderr, "Invalid --max-steps value: %s\n", argv[i]);
                    cli_status = 1;
                    break;
                }
            } else if (std::strcmp(argv[i], "--max-ray-attempts") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires an integer of at least 3\n",
                                 argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (!parse_int(argv[i], 3, &max_ray_attempts)) {
                    std::fprintf(stderr, "Invalid --max-ray-attempts value: %s\n",
                                 argv[i]);
                    cli_status = 1;
                    break;
                }
            } else if (std::strcmp(argv[i], "--alpha") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires a positive finite number\n",
                                 argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (!parse_positive_double(argv[i], &alpha)) {
                    std::fprintf(stderr, "Invalid --alpha value: %s\n", argv[i]);
                    cli_status = 1;
                    break;
                }
                alpha_provided = true;
            } else if (std::strcmp(argv[i], "--source-mode") == 0) {
                if (++i >= argc) {
                    std::fprintf(stderr, "%s requires uniform or green\n", argv[i-1]);
                    cli_status = 1;
                    break;
                }
                if (std::strcmp(argv[i], "uniform") == 0) {
                    source_mode = SourceMode::Uniform;
                } else if (std::strcmp(argv[i], "green") == 0) {
                    source_mode = SourceMode::Green;
                } else {
                    std::fprintf(stderr, "Invalid --source-mode value: %s\n", argv[i]);
                    cli_status = 1;
                    break;
                }
                source_mode_provided = true;
            } else if (std::strcmp(argv[i], "--force") == 0) {
                force_output = true;
            } else if (argv[i][0] == '-') {
                std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
                cli_status = 1;
                break;
            } else {
                positional.push_back(argv[i]);
            }
        }

        if (cli_status == 0 && positional.size() > 5) {
            std::fprintf(stderr, "Too many positional arguments\n");
            cli_status = 1;
        }
        if (cli_status == 0 && positional.size() == 2) {
            std::fprintf(stderr, "Nx and Ny must be provided together\n");
            cli_status = 1;
        }
        if (cli_status == 0 && !positional.empty()) eq_name = positional[0];
        if (cli_status == 0 && positional.size() >= 3) {
            if (!parse_int(positional[1], 1, &Nx) ||
                !parse_int(positional[2], 1, &Ny)) {
                std::fprintf(stderr, "Nx and Ny must be positive integers\n");
                cli_status = 1;
            }
        }
        if (cli_status == 0 && positional.size() >= 4) {
            // Try parse the fourth positional argument as Nz; otherwise it is the mesh.
            int parsed_nz = 0;
            if (parse_int(positional[3], 1, &parsed_nz)) {
                Nz = parsed_nz;
                if (positional.size() == 5) mesh_filename = positional[4];
            } else {
                if (positional.size() == 5) {
                    std::fprintf(stderr, "Nz must be a positive integer when a fifth positional argument is present\n");
                    cli_status = 1;
                } else {
                    mesh_filename = positional[3];
                }
            }
        }

        if (cli_status == 0 && !seed_provided) {
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            seed = static_cast<std::uint64_t>(now);
        }
    }

    MPI_Bcast(&cli_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (cli_status != 0) {
        MPI_Finalize();
        return cli_status == 2 ? 0 : 1;
    }

    MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_walks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_ray_attempts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int source_mode_value = static_cast<int>(source_mode);
    MPI_Bcast(&source_mode_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    source_mode = static_cast<SourceMode>(source_mode_value);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Random seed: %llu%s\n",
                    static_cast<unsigned long long>(seed),
                    seed_provided ? " (fixed by --seed)" : " (generated from current time)");
    }

    int eq_idx = -1;
    if (rank == 0) {
        for (int i = 0; i < (int)(sizeof(equation_registry) / sizeof(*equation_registry)); i++) {
            if (std::strcmp(equation_registry[i]->name, eq_name) == 0) {
                eq_idx = i;
                break;
            }
        }
        if (eq_idx < 0) {
            std::fprintf(stderr, "Unknown equation: '%s'\nAvailable equations:\n", eq_name);
            for (const auto *e : equation_registry) {
                std::fprintf(stderr, "  - %s\n", e->name);
            }
        }
    }
    MPI_Bcast(&eq_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (eq_idx < 0) {
        MPI_Finalize();
        return 1;
    }
    const Equation *eq = equation_registry[eq_idx];

    if (rank == 0 && eq != &screened_poisson) {
        if (alpha_provided) {
            std::fprintf(stderr,
                         "Warning: --alpha applies only to screened_poisson; "
                         "it will be ignored for equation '%s'.\n",
                         eq->name);
        }
        if (source_mode_provided) {
            std::fprintf(stderr,
                         "Warning: --source-mode applies only to screened_poisson; "
                         "it will be ignored for equation '%s'.\n",
                         eq->name);
        }
    }

    int output_ok = 1;
    if (rank == 0 && !force_output) {
        std::error_code error;
        bool exists = std::filesystem::exists(output_filename, error);
        if (error) {
            std::fprintf(stderr, "Could not inspect output path %s: %s\n",
                         output_filename, error.message().c_str());
            output_ok = 0;
        } else if (exists) {
            std::fprintf(stderr,
                         "Refusing to overwrite existing output: %s\n"
                         "Choose another path with --output, or pass --force.\n",
                         output_filename);
            output_ok = 0;
        }
    }
    MPI_Bcast(&output_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!output_ok) {
        MPI_Finalize();
        return 1;
    }

    // broadcast mesh filename
    int mesh_filename_len = (rank == 0) ? (int)std::strlen(mesh_filename) + 1 : 0;
    MPI_Bcast(&mesh_filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> mesh_filename_buf(mesh_filename_len);
    if (rank == 0) std::memcpy(mesh_filename_buf.data(), mesh_filename, mesh_filename_len);
    MPI_Bcast(mesh_filename_buf.data(), mesh_filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // broadcast output filename
    int output_filename_len = (rank == 0) ? (int)std::strlen(output_filename) + 1 : 0;
    MPI_Bcast(&output_filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> output_filename_buf(output_filename_len);
    if (rank == 0) std::memcpy(output_filename_buf.data(), output_filename, output_filename_len);
    MPI_Bcast(output_filename_buf.data(), output_filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // peek mesh dim, dispatch to equation's 2D or 3D entrypoint
    int dim = 0;
    int mesh_dim_ok = 1;
    if (rank == 0) {
        try {
            dim = peek_mesh_dim(mesh_filename_buf.data());
        } catch (const std::exception &error) {
            std::fprintf(stderr, "%s\n", error.what());
            mesh_dim_ok = 0;
        }
    }
    MPI_Bcast(&mesh_dim_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!mesh_dim_ok) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rc;
    if (dim == 2) {
        rc = eq->run_2D(rank, size, mesh_filename_buf.data(), output_filename_buf.data(),
                        Nx, Ny, Nz, N_walks, epsilon, max_steps,
                        max_ray_attempts, seed, alpha, source_mode);
    } else {
        rc = eq->run_3D(rank, size, mesh_filename_buf.data(), output_filename_buf.data(),
                        Nx, Ny, Nz, N_walks, epsilon, max_steps,
                        max_ray_attempts, seed, alpha, source_mode);
    }

    MPI_Finalize();
    return rc;
}
