cmake_minimum_required(VERSION 3.20)
project(wos CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / intellisense

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# core library
add_library(wos_core STATIC
    src/bvh.cpp
    src/wos.cpp
    src/mesh.cpp
    src/inside.cpp
    src/npq.cpp
    src/prng.cpp
)
target_include_directories(wos_core PUBLIC include ${HDF5_INCLUDE_DIRS})
target_link_libraries(wos_core PUBLIC MPI::MPI_CXX ${HDF5_LIBRARIES})

# application: launcher + equation registry + each equation in its own TU
add_executable(wos
    src/main.cpp
    src/equations/laplace.cpp
    src/equations/poisson.cpp
    src/equations/screened_poisson.cpp
    src/equations/helmholtz.cpp
)
target_include_directories(wos PRIVATE src)
target_link_libraries(wos PRIVATE wos_core)

# Standalone transient-heat experiments for welding applications.  This is
# intentionally kept separate from the original steady-state `wos` program.
option(WOS_BUILD_WELDING "Build the welding transient-heat experiments" ON)
if(WOS_BUILD_WELDING)
    add_subdirectory(apps/welding)
endif()
