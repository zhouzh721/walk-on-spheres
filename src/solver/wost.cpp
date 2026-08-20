#include "wos/solver/wost.hpp"

namespace wos::solver {

WoSt::WoSt(Settings settings) : settings_(settings) {
    validate(settings_);
}

} // namespace wos::solver
