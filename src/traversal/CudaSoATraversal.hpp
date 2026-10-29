#pragma once

#include <cuda_runtime.h>

#include "../container/CudaSoAContainer.hpp"
#include "../particle/Particle.hpp"
#include "../util/MathUtils.hpp"

/**
 * SoA CUDA kernel for applying a functor to each particle in the container.
 * Materialises the particle from the SoA buffers, calls the functor and writes
 * it back. Can be used e.g. for integration and boundaries.
 */
template <typename Func>
__global__ void cudaForEachSoAKernel(CudaSoA p, uint_type n, Func func) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle particle_i;
    particle_i.id = p.id[i];
    particle_i.r = {p.rx[i], p.ry[i], p.rz[i]};
    particle_i.v = {p.vx[i], p.vy[i], p.vz[i]};
    particle_i.f = {p.fx[i], p.fy[i], p.fz[i]};
    func(particle_i);
    p.id[i] = particle_i.id;
    p.rx[i] = particle_i.r[0];
    p.ry[i] = particle_i.r[1];
    p.rz[i] = particle_i.r[2];
    p.vx[i] = particle_i.v[0];
    p.vy[i] = particle_i.v[1];
    p.vz[i] = particle_i.v[2];
    p.fx[i] = particle_i.f[0];
    p.fy[i] = particle_i.f[1];
    p.fz[i] = particle_i.f[2];
}

/**
 * SoA CUDA kernel for computing pairwise forces between particles in the container.
 * Forces are accumulated locally and written to fx/fy/fz at index i.
 *
 * Does not use Newton's third law: each (i,j) pair is computed twice,
 * so the thread owns the complete force on i and writes only its own entry (no atomics needed).
 */
template <typename Func>
__global__ void cudaForceSoAKernel(CudaSoA p, uint_type n, const uint_type* bin_offsets,
                                CellIndex cell_indexing, float_type interaction_radius_sq, Func func) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle particle_i;
    particle_i.id = p.id[i];
    particle_i.r = {p.rx[i], p.ry[i], p.rz[i]};
    particle_i.v = {p.vx[i], p.vy[i], p.vz[i]};
    particle_i.f = {p.fx[i], p.fy[i], p.fz[i]};

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
            Particle particle_j;
            particle_j.id = p.id[j];
            particle_j.r = {p.rx[j], p.ry[j], p.rz[j]};
            particle_j.v = {p.vx[j], p.vy[j], p.vz[j]};
            particle_j.f = {p.fx[j], p.fy[j], p.fz[j]};
            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
            const Vec3 f = func(particle_i, particle_j);
            force[0] += f[0];
            force[1] += f[1];
            force[2] += f[2];
        }
    }

    p.fx[i] += force[0];
    p.fy[i] += force[1];
    p.fz[i] += force[2];
}

/**
 * CUDA particle-based traversal on a struct-of-arrays container.
 */
class CudaSoATraversal {
    CudaSoAContainer* _container;

   public:
    CudaSoATraversal(CudaSoAContainer* container) : _container(container) {}

    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        const uint_type num_particles = _container->size();
        if (num_particles > 0) {
            cudaForEachSoAKernel<<<cudaNumBlocks(num_particles), CUDA_THREADS_PER_BLOCK>>>(
                _container->_soa, num_particles, func);
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
            cudaForceSoAKernel<<<cudaNumBlocks(num_particles), CUDA_THREADS_PER_BLOCK>>>(
                _container->_soa, num_particles, bin_offsets, cell_indexing, interaction_radius_sq, func);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        _container->modifyDevice();
    }
};
