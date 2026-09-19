#pragma once

#include <cstddef>
#include <limits>

namespace wowee {

/// Megabytes to bytes, saturating at SIZE_MAX rather than wrapping. On a 32-bit
/// target (wasm32) the 4 and 8 GB cache budgets overflow size_t, and a budget
/// that wraps to 0 evicts every texture the moment it is cached.
constexpr size_t mbToBytes(size_t mb) {
    constexpr size_t kMB = 1024 * 1024;
    return mb > std::numeric_limits<size_t>::max() / kMB
        ? std::numeric_limits<size_t>::max()
        : mb * kMB;
}

} // namespace wowee
