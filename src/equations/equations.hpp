#pragma once
#include <stdint.h>

struct Equation {
    const char *name;
    int (*run_2D)(int rank, int size, const char *mesh_filename, const char *output_filename,
                  int Nx, int Ny, int Nz, int N_walks, double epsilon,
                  int max_steps, int max_ray_attempts, uint64_t seed);
    int (*run_3D)(int rank, int size, const char *mesh_filename, const char *output_filename,
                  int Nx, int Ny, int Nz, int N_walks, double epsilon,
                  int max_steps, int max_ray_attempts, uint64_t seed);
};

// link equations
extern const Equation laplace;
extern const Equation poisson;
extern const Equation screened_poisson;
extern const Equation helmholtz;

// equation registry
inline const Equation *const equation_registry[] = {
    &laplace,
    &poisson,
    &screened_poisson, 
    &helmholtz,
};
