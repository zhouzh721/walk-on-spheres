#include "wos/solver/wost.hpp"

namespace wos::solver {

WoSt::WoSt(Settings settings)
    : settings_(WoStSettings{settings, 1e-8, 0.99, 4}) {
    validate(settings_);
}

WoSt::WoSt(WoStSettings settings) : settings_(settings) {
    validate(settings_);
}

} // namespace wos::solver
