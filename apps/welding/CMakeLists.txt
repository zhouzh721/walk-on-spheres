add_executable(wos_welding_level0
    src/level0_cooling.cpp
)

add_executable(wos_welding_level1
    src/level1_stationary_heat.cpp
)

foreach(welding_target IN ITEMS wos_welding_level0 wos_welding_level1)
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
