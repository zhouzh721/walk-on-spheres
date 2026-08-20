add_executable(wos_welding_level0
    src/level0_cooling.cpp
)

add_executable(wos_welding_level1
    src/level1_stationary_heat.cpp
)

add_executable(wos_welding_level2
    src/level2_transient_heat.cpp
)

add_executable(wos_welding_level3
    src/level3_moving_heat.cpp
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(wos_welding_level2 PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(wos_welding_level2
        PRIVATE WELDING_LEVEL2_HAS_OPENMP=1
    )
    target_link_libraries(wos_welding_level3 PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(wos_welding_level3
        PRIVATE WELDING_LEVEL3_HAS_OPENMP=1
    )
endif()

foreach(welding_target IN ITEMS
        wos_welding_level0
        wos_welding_level1
        wos_welding_level2
        wos_welding_level3)
    target_include_directories(${welding_target}
        PRIVATE include
    )

    target_compile_definitions(${welding_target}
        PRIVATE WELDING_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
    )

    target_link_libraries(${welding_target}
        PRIVATE wos_core
    )
endforeach()
