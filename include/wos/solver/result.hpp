#pragma once

#include "wos/solver/termination.hpp"

namespace wos::solver {

struct SampleResult {
    double value;
    int steps;
    TerminationReason termination;
};

struct Result {
    double mean;
    double variance;
    double standard_error;
    double mean_steps;
    int max_steps_hits;
    int invalid_paths = 0;
};

} // namespace wos::solver
