#pragma once

namespace wos::solver {

struct Settings {
    int walks;
    double epsilon;
    int max_steps;
};

void validate(const Settings &settings);

struct WoStSettings {
    Settings walk;
    double minimum_radius = 1e-8;
    double radius_shrink = 0.99;
    int max_retries = 4;
};

void validate(const WoStSettings &settings);

} // namespace wos::solver
