#pragma once

namespace wos::solver {

struct Settings {
    int walks;
    double epsilon;
    int max_steps;
};

void validate(const Settings &settings);

} // namespace wos::solver
