#pragma once

namespace wos::solver {

struct SampleResult {
    double value;
    int steps;
    bool max_steps_reached;
};

struct Result {
    double mean;
    double variance;
    double standard_error;
    double mean_steps;
    int max_steps_hits;
};

} // namespace wos::solver
