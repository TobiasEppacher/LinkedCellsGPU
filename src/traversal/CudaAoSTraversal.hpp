#pragma once

#include <cuda_runtime.h>

#include "../container/CudaAoSContainer.hpp"
#include "../particle/Particle.hpp"
#include "../util/MathUtils.hpp"

/**
 * AoS CUDA kernel for applying a functor to each particle in the container.
 * Mutates particles in place. Can be used e.g. for integration and boundaries.
 */
template <typename Func>
__global__ void cudaForEachAoSKernel(CudaAoS p, uint_type n, Func func) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    func(p.particles[i]);
}

/**
 * AoS CUDA kernel for computing pairwise force between particles in the container.
 * Forces are accumulated per locally and written to the `f` member of each particle.
 *
 * Does not use Newton's third law: each (i,j) pair is computed twice,
 * so the thread owns the complete force on i and writes only its f member (no atomics needed).
 */
template <typename Func>
__global__ void cudaForceAoSKernel(CudaAoS p, uint_type n, const uint_type* bin_offsets,
                                   CellIndex cell_indexing, float_type interaction_radius_sq, Func func) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Particle particle_i = p.particles[i];

    const Vec3u cell_coords_i = cell_indexing.decode(p.cell_idx[i]);

    Vec3 force = {float_type(0), float_type(0), float_type(0)};
    for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
    for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
    for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
        const uint_type cell_idx_j = cell_indexing.encode({
            static_cast<uint_type>(cell_coords_i[0] + delta_x),
            static_cast<uint_type>(cell_coords_i[1] + delta_y),
            static_cast<uint_type>(cell_coords_i[2] + delta_z)});
        const uint_type j_begin = bin_offsets[cell_idx_j];
        const uint_type j_end = bin_offsets[cell_idx_j + 1];
        for (uint_type j = j_begin; j < j_end; ++j) {
            if (j == i) continue;
            const Particle particle_j = p.particles[j];
            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
            const Vec3 f = func(particle_i, particle_j);
            force[0] += f[0];
            force[1] += f[1];
            force[2] += f[2];
        }
    }

    p.particles[i].f[0] += force[0];
    p.particles[i].f[1] += force[1];
    p.particles[i].f[2] += force[2];
}

/**
 * CUDA particle-based traversal on an array-of-structs (AoS) container.
 * The `applyForEach` and `applyPairwise` methods are conceptually indentical
 * to the `CudaSoATraversal`, but use the AoS memory layout in the
 * corresponding CUDA kernels instead.
 */
class CudaAoSTraversal {
    CudaAoSContainer* _container;

   public:
    CudaAoSTraversal(CudaAoSContainer* container) : _container(container) {}

    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        const uint_type num_particles = _container->size();
        if (num_particles > 0) {
            cudaForEachAoSKernel<<<cudaNumBlocks(num_particles), CUDA_THREADS_PER_BLOCK>>>(
                _container->_aos, num_particles, func);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        _container->modifyDevice();
    }

    // Requires an up-to-date cell binning: call the container's rebuild() first.
    template <typename Func>
    void applyPairwise(Func func) {
        const uint_type num_particles = _container->size();
        const uint_type* bin_offsets = _container->getBinOffsets();
        const float_type interaction_radius_sq =
            _container->getInteractionRadius() * _container->getInteractionRadius();
        const CellIndex cell_indexing = _container->_cell_indexing;

        if (num_particles > 0) {
            cudaForceAoSKernel<<<cudaNumBlocks(num_particles), CUDA_THREADS_PER_BLOCK>>>(
                _container->_aos, num_particles, bin_offsets, cell_indexing, interaction_radius_sq, func);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        _container->modifyDevice();
    }
};
