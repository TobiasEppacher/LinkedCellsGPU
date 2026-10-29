#pragma once

#include <cstddef>
#include <iostream>

#include "KokkosDefinitions.hpp"

/**
 * Returns the team scratch level for a per-team request of `scratch_bytes`: level 0
 * (on-chip shared memory) if it fits Kokkos' limit, else level 1, which is backed by
 * global memory on GPUs. The fallback is reported once per `warned` flag.
 */
inline int selectScratchLevel(size_t scratch_bytes, const char* kernel_name, bool& warned) {
    const size_t max_l0_bytes = static_cast<size_t>(Kokkos::TeamPolicy<Device>::scratch_size_max(0));
    if (scratch_bytes <= max_l0_bytes) return 0;
    if (!warned) {
        std::cerr << "Warning: " << kernel_name << " needs " << scratch_bytes
                  << " B of team scratch, above the level-0 limit of " << max_l0_bytes
                  << " B. Falling back to level-1 scratch (global memory on GPUs).\n";
        warned = true;
    }
    return 1;
}
