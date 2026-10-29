#pragma once

#include <cuda_runtime.h>

#include "../util/TypeDefinitions.hpp"

/**
 * Launch geometry and the two sorting kernels that touch nothing but the cell index,
 * shared by CudaSoAContainer and CudaAoSContainer. Everything that reads or moves
 * particle data is layout-specific and lives in the containers themselves.
 */

/**
 * Threads per block for all one-thread-per-particle kernels. 128 matches what
 * Kokkos picks for its RangePolicy: the per-particle work is uneven, and smaller
 * blocks let the scheduler rebalance the more finely.
 */
static constexpr unsigned int CUDA_THREADS_PER_BLOCK = 128;

// Number of blocks needed to cover n elements with CUDA_THREADS_PER_BLOCK threads each.
inline unsigned int cudaNumBlocks(uint_type n) {
    return static_cast<unsigned int>((n + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
}

/**
 * atomicAdd instruction for the different uint_type depending on the bitsize selected in TypeDefinitions.hpp
 */
__device__ __forceinline__ uint_type atomicAddUint(uint_type* address, uint_type value) {
#if defined(LC_USE_64_BIT) && LC_USE_64_BIT
    return static_cast<uint_type>(
        atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
#else
    return atomicAdd(reinterpret_cast<unsigned int*>(address), static_cast<unsigned int>(value));
#endif
}

/**
 * Kernel for counting the number of particles in each cell
 * (TODO: Check for better implementation without atomicAdd)
 */
__global__ inline void cudaCountCellsKernel(const uint_type* cell_idx, uint_type num_particles, uint_type* cell_counts) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    atomicAddUint(&cell_counts[cell_idx[i]], uint_type(1));
}

/**
 * Scatter the particles (by index) into the cells respective ranges
 * Creates a permutation array that maps the new particle order to the old one
 */
__global__ inline void cudaScatterPermKernel(const uint_type* cell_idx, uint_type num_particles, uint_type* cursor, uint_type* perm) {
    const uint_type old_i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (old_i >= num_particles) return;
    const uint_type new_i = atomicAddUint(&cursor[cell_idx[old_i]], uint_type(1));
    perm[new_i] = old_i;
}
