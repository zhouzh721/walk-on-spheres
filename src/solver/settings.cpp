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

void validate(const WoStSettings &settings) {
    validate(settings.walk);
    if (!(settings.minimum_radius > 0.0) ||
        !std::isfinite(settings.minimum_radius)) {
        throw std::invalid_argument(
            "minimum WoSt radius must be finite and positive");
    }
    if (!(settings.radius_shrink > 0.0 && settings.radius_shrink < 1.0)) {
        throw std::invalid_argument(
            "WoSt radius shrink must lie strictly between zero and one");
    }
    if (settings.max_retries < 0) {
        throw std::invalid_argument("WoSt retry count cannot be negative");
    }
}

} // namespace wos::solver
