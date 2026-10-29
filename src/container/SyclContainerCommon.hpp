#pragma once

#include <sycl/sycl.hpp>

#include "../util/TypeDefinitions.hpp"

/**
 * Launch geometry and the device atomic shared by SyclSoAContainer and SyclAoSContainer.
 * Everything that reads or moves particle data is layout-specific and lives in the
 * containers themselves.
 */

// Work-group size for all one-work-item-per-particle kernels, matching
// CUDA_THREADS_PER_BLOCK so both backends launch the same shape.
static constexpr std::size_t SYCL_WORK_GROUP_SIZE = 128;

// Global range rounded up to a whole number of work groups; kernels guard with i >= n.
inline std::size_t syclGlobalSize(uint_type n) {
    return ((static_cast<std::size_t>(n) + SYCL_WORK_GROUP_SIZE - 1) / SYCL_WORK_GROUP_SIZE) * SYCL_WORK_GROUP_SIZE;
}

// Number of work groups a scan of `len` elements splits into.
inline uint_type numScanGroups(uint_type len) {
    return static_cast<uint_type>((static_cast<std::size_t>(len) + SYCL_WORK_GROUP_SIZE - 1) /
                                  SYCL_WORK_GROUP_SIZE);
}

// Device-side atomic increment on uint_type (uint64_t or uint32_t per TypeDefinitions.hpp).
using SyclUintAtomicRef = sycl::atomic_ref<
    uint_type,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
>;
