#include "wos/solver/wos.hpp"

namespace wos::solver {

WoS::WoS(Settings settings) : settings_(settings) {
    validate(settings_);
}

} // namespace wos::solver
