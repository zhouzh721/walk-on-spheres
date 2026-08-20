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

# FCPW is the default geometry-query backend. Keep the first integration
# CPU-only and scalar so the dependency remains small and predictable.
set(FCPW_USE_ENOKI OFF CACHE BOOL "Build enoki" FORCE)
set(FCPW_ENABLE_GPU_SUPPORT OFF CACHE BOOL "Enable GPU support" FORCE)
set(FCPW_BUILD_BINDINGS OFF CACHE BOOL "Build bindings" FORCE)
set(FCPW_BUILD_DEMO OFF CACHE BOOL "Build demo" FORCE)
set(FCPW_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fcpw/CMakeLists.txt OR
   NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fcpw/deps/eigen/Eigen/Core)
    message(FATAL_ERROR
        "FCPW is not initialized. Run: git submodule update --init --recursive")
endif()
add_subdirectory(third_party/fcpw EXCLUDE_FROM_ALL)
target_include_directories(fcpw SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fcpw/deps/eigen)

# core library
add_library(wos_core STATIC
    src/boundary/fcpw_scene.cpp
    src/boundary/scene.cpp
    src/geometry/classification.cpp
    src/geometry/fcpw_scene.cpp
    src/mesh.cpp
    src/prng.cpp
    src/sampling/uniform.cpp
    src/sampling/neumann_boundary.cpp
    src/solver/settings.cpp
    src/solver/wos.cpp
    src/solver/wost.cpp
)
target_include_directories(wos_core PUBLIC include ${HDF5_INCLUDE_DIRS})
target_link_libraries(wos_core
    PUBLIC MPI::MPI_CXX ${HDF5_LIBRARIES}
    PRIVATE fcpw)
set_source_files_properties(
    src/boundary/fcpw_scene.cpp
    src/geometry/fcpw_scene.cpp
    PROPERTIES
    COMPILE_OPTIONS
        "-Wno-unused-parameter;-Wno-unused-variable;-Wno-sign-compare;-Wno-reorder")

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

include(CTest)
if(BUILD_TESTING)
    add_executable(wos_test_bessel tests/test_bessel.cpp)
    target_link_libraries(wos_test_bessel PRIVATE wos_core)
    add_test(NAME wos_test_bessel COMMAND wos_test_bessel)

    add_executable(wos_test_screened tests/test_screened.cpp)
    target_link_libraries(wos_test_screened PRIVATE wos_core)
    add_test(NAME wos_test_screened COMMAND wos_test_screened)

    add_executable(wos_test_boundary tests/test_boundary.cpp)
    target_link_libraries(wos_test_boundary PRIVATE wos_core)
    add_test(NAME wos_test_boundary COMMAND wos_test_boundary)

    add_executable(wos_test_boundary_scene tests/test_boundary_scene.cpp)
    target_link_libraries(wos_test_boundary_scene PRIVATE wos_core)
    add_test(NAME wos_test_boundary_scene COMMAND wos_test_boundary_scene)

    add_executable(wos_test_fcpw_geometry_scene
        tests/test_fcpw_geometry_scene.cpp)
    target_link_libraries(wos_test_fcpw_geometry_scene PRIVATE wos_core)
    add_test(NAME wos_test_fcpw_geometry_scene
        COMMAND wos_test_fcpw_geometry_scene)

    add_executable(wos_test_wost tests/test_wost.cpp)
    target_link_libraries(wos_test_wost PRIVATE wos_core)
    add_test(NAME wos_test_wost COMMAND wos_test_wost)

    add_executable(wos_test_wost_source tests/test_wost_source.cpp)
    target_link_libraries(wos_test_wost_source PRIVATE wos_core)
    add_test(NAME wos_test_wost_source COMMAND wos_test_wost_source)

    add_executable(wos_test_termination tests/test_termination.cpp)
    target_link_libraries(wos_test_termination PRIVATE wos_core)
    add_test(NAME wos_test_termination COMMAND wos_test_termination)
endif()

# Standalone transient-heat experiments for welding applications.  This is
# intentionally kept separate from the original steady-state `wos` program.
option(WOS_BUILD_WELDING "Build the welding transient-heat experiments" ON)
if(WOS_BUILD_WELDING)
    add_subdirectory(apps/welding)
endif()
