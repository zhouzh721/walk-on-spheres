#pragma once

namespace wos {

// Result produced by one walk-on-spheres path.
struct SampleResult {
    double value;
    int steps;
    bool max_steps_reached;
};

// Statistics collected from multiple walk-on-spheres paths.
struct WoSResult {
    double mean;
    double variance;
    double standard_error;
    double mean_steps;
    int max_steps_hits;
};

}
