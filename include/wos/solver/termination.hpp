#pragma once

namespace wos::solver {

enum class TerminationReason {
    ReachedDirichlet,
    ReachedMaxSteps,
    Invalid,
};

} // namespace wos::solver
