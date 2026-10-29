#pragma once

#include <cmath>

/**
 * Marks a function as callable from device kernels, whichever backend compiles
 * this translation unit:
 *   clang/nvcc in CUDA mode (Kokkos CUDA build, native CUDA build) -> __host__ __device__
 *   acpp in the generic (SSCP) flow                                -> plain inline
 */
#if defined(__CUDACC__) || defined(__CUDA__)
#define LC_HOST_DEVICE_INLINE __host__ __device__ inline
#else
#define LC_HOST_DEVICE_INLINE inline
#endif

/**
 * Math used inside kernels. std:: resolves to the right device builtins under
 * both clang-CUDA and AdaptiveCpp, keeping the force functors backend-agnostic.
 */
namespace lc {
using std::sqrt;
}  // namespace lc
