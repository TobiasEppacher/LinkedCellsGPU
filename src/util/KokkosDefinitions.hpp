#pragma once

#include <Kokkos_Core.hpp>

/**
 * Execution-space aliases for the Kokkos-based containers and traversals. Kept
 * out of TypeDefinitions.hpp so the Kokkos-free code (the SYCL binary, io/, functors/)
 * can use the scalar typedefs without pulling in Kokkos.
 */
using Serial = Kokkos::Serial;
using Host = Kokkos::DefaultHostExecutionSpace;
using Device = Kokkos::DefaultExecutionSpace;

// Upper bound for explicit team sizes (CUDA's max threads per block)
inline constexpr int MAX_TEAM_SIZE = 1024;
