add_executable(wos_welding_level0
    src/level0_cooling.cpp
)

target_include_directories(wos_welding_level0
    PRIVATE include
)

target_compile_definitions(wos_welding_level0
    PRIVATE WELDING_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
)

target_link_libraries(wos_welding_level0
    PRIVATE wos_core
)
