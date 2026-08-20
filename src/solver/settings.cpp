#include <cmath>
#include <stdexcept>

#include "wos/solver/settings.hpp"

namespace wos::solver {

void validate(const Settings &settings) {
    if (settings.walks <= 0) {
        throw std::invalid_argument("solver walk count must be positive");
    }
    if (!(settings.epsilon > 0.0) || !std::isfinite(settings.epsilon)) {
        throw std::invalid_argument("solver epsilon must be finite and positive");
    }
    if (settings.max_steps <= 0) {
        throw std::invalid_argument("solver maximum step count must be positive");
    }
}

} // namespace wos::solver
