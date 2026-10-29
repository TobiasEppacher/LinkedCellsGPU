#pragma once

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <stdexcept>
#include <vector>

#include "CudaContainerCommon.hpp"
#include "ParticleContainer.hpp"
#include "../traversal/CellIndexing.hpp"
#include "../util/CudaCheck.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Native-CUDA counterpart of KokkosSoAContainer. It stores the particle data in
 * the same structure-of-arrays layout and offers the same rebuild interface
 * (updateCellIdx + sortByCellIdx producing a per-cell bin_offsets array), but
 * uses raw cudaMalloc'd device buffers and hand-written kernels instead of
 * Kokkos Views and parallel_for. The host mirror is kept in std::vectors and copied
 * to/from the device explicitly, with a single host/device dirty flag pair (all
 * arrays are always modified together in this codebase).
 */

/**
 * Bundle of raw device pointers describing the SoA particle buffers.
 * Passed by value into kernels (analogous to capturing the individual Kokkos device views)
 */
struct CudaSoA {
    uint_type* cell_idx = nullptr;
    uint_type* id = nullptr;
    float_type* rx = nullptr;
    float_type* ry = nullptr;
    float_type* rz = nullptr;
    float_type* vx = nullptr;
    float_type* vy = nullptr;
    float_type* vz = nullptr;
    float_type* fx = nullptr;
    float_type* fy = nullptr;
    float_type* fz = nullptr;
};


// --- Cell update kernel ---

/**
 * Kernel for recomputation of the cell indices of all particles based on their current positions.
 */
__global__ void cudaUpdateCellIdxKernel(CudaSoA p, uint_type num_particles, Vec3 box_min_with_halo,
                                        Vec3 cell_size, CellIndex cell_indexing) {
    const uint_type i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    const uint_type cx = static_cast<uint_type>((p.rx[i] - box_min_with_halo[0]) / cell_size[0]);
    const uint_type cy = static_cast<uint_type>((p.ry[i] - box_min_with_halo[1]) / cell_size[1]);
    const uint_type cz = static_cast<uint_type>((p.rz[i] - box_min_with_halo[2]) / cell_size[2]);
    p.cell_idx[i] = cell_indexing.encode({cx, cy, cz});
}

// --- Sorting kernels ---

/**
 * Reorder the particle data according to the permutation array
 * dst[new_i] = src[old_i] where old_i = perm[new_i]
 */
__global__ void cudaApplyPermutationSoAKernel(CudaSoA dst, CudaSoA src, const uint_type* perm, uint_type num_particles) {
    const uint_type new_i = static_cast<uint_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (new_i >= num_particles) return;
    const uint_type old_i = perm[new_i];
    dst.cell_idx[new_i] = src.cell_idx[old_i];
    dst.id[new_i] = src.id[old_i];
    dst.rx[new_i] = src.rx[old_i];
    dst.ry[new_i] = src.ry[old_i];
    dst.rz[new_i] = src.rz[old_i];
    dst.vx[new_i] = src.vx[old_i];
    dst.vy[new_i] = src.vy[old_i];
    dst.vz[new_i] = src.vz[old_i];
    dst.fx[new_i] = src.fx[old_i];
    dst.fy[new_i] = src.fy[old_i];
    dst.fz[new_i] = src.fz[old_i];
}

// --- Container class ---

class CudaSoAContainer : ParticleContainer {
    friend class CudaSoATraversal;

    // Host mirror (structure-of-arrays); always sized to _num_particles.
    std::vector<uint_type> _h_cell_idx;
    std::vector<uint_type> _h_id;
    std::vector<float_type> _h_rx, _h_ry, _h_rz;
    std::vector<float_type> _h_vx, _h_vy, _h_vz;
    std::vector<float_type> _h_fx, _h_fy, _h_fz;

    // Live device particle buffers
    CudaSoA _soa;

    // Device buffer for out-of-place reordering (temporary)
    CudaSoA _reorder_tmp_soa;

    // Permutation array for reordering
    uint_type* _d_perm = nullptr;

    // Allocated capacity of _soa and _reorder_tmp_soa in particles
    uint_type _dev_capacity = 0;

    // Per-cell start offsets into the particle array
    uint_type* _d_bin_offsets = nullptr;

    // Per-cell write cursor for scattering into the particle array (temporary)
    uint_type* _d_bin_cursor = nullptr;

    // Dirty flag for the host mirror
    bool _host_modified = false;

    // Dirty flag for the device buffers
    bool _device_modified = false;

    Vec3 _box_min;
    Vec3 _box_max;
    float_type _interaction_radius;
    uint_type _num_particles;

    Vec3u _num_cells_with_halo;
    Vec3 _cell_size;
    Vec3 _box_min_with_halo;
    Vec3 _box_max_with_halo;
    uint_type _num_cells;

    CellIndex _cell_indexing;

   public:
    CudaSoAContainer(Vec3 box_min, Vec3 box_max, float_type interaction_radius)
        : _box_min(box_min), 
          _box_max(box_max), 
          _interaction_radius(interaction_radius), 
          _num_particles(0) {
        _num_cells_with_halo = {static_cast<uint_type>((_box_max[0] - _box_min[0]) / _interaction_radius) + 2,
                                static_cast<uint_type>((_box_max[1] - _box_min[1]) / _interaction_radius) + 2,
                                static_cast<uint_type>((_box_max[2] - _box_min[2]) / _interaction_radius) + 2};
        _cell_size = {(_box_max[0] - _box_min[0]) / (_num_cells_with_halo[0] - 2),
                      (_box_max[1] - _box_min[1]) / (_num_cells_with_halo[1] - 2),
                      (_box_max[2] - _box_min[2]) / (_num_cells_with_halo[2] - 2)};
       _box_min_with_halo = {_box_min[0] - _cell_size[0], _box_min[1] - _cell_size[1],
                              _box_min[2] - _cell_size[2]};
        _box_max_with_halo = {_box_max[0] + _cell_size[0], _box_max[1] + _cell_size[1],
                              _box_max[2] + _cell_size[2]};

        _num_cells = _num_cells_with_halo[0] * _num_cells_with_halo[1] * _num_cells_with_halo[2];
        _cell_indexing = CellIndex(_num_cells_with_halo);

        CUDA_CHECK(cudaMalloc(&_d_bin_offsets, (_num_cells + 1) * sizeof(uint_type)));
        CUDA_CHECK(cudaMalloc(&_d_bin_cursor, _num_cells * sizeof(uint_type)));
    }

    // Raw device ownership makes copying unsafe; the container is only ever used
    // by value in place (never copied), matching how the Kokkos container is used.
    CudaSoAContainer(const CudaSoAContainer&) = delete;
    CudaSoAContainer& operator=(const CudaSoAContainer&) = delete;

    ~CudaSoAContainer() override {
        freeSoA(_soa);
        freeSoA(_reorder_tmp_soa);
        cudaFree(_d_perm);
        cudaFree(_d_bin_offsets);
        cudaFree(_d_bin_cursor);
    }

    // Host-side addition of a particle
    void addParticle(const Particle& p) override {
        _h_cell_idx.push_back(getCellIndex(p.r));
        _h_id.push_back(p.id);
        _h_rx.push_back(p.r[0]);
        _h_ry.push_back(p.r[1]);
        _h_rz.push_back(p.r[2]);
        _h_vx.push_back(p.v[0]);
        _h_vy.push_back(p.v[1]);
        _h_vz.push_back(p.v[2]);
        _h_fx.push_back(p.f[0]);
        _h_fy.push_back(p.f[1]);
        _h_fz.push_back(p.f[2]);
        ++_num_particles;
        modifyHost();
    }

    uint_type size() const override { return _num_particles; }

    void rebuild() override {
        updateCellIdx();
        sortByCellIdx();
    }

    // --- Host/Device sync semantics ---

    void modifyHost() { _host_modified = true; }
    void modifyDevice() { _device_modified = true; }

    // Copy the host mirror to the device if the host side is dirty.
    void syncDevice() {
        if (!_host_modified) return;
        ensureDeviceCapacity(_num_particles);
        const uint_type n = _num_particles;
        if (n > 0) {
            copyToDevice(_soa.cell_idx, _h_cell_idx);
            copyToDevice(_soa.id, _h_id);
            copyToDevice(_soa.rx, _h_rx);
            copyToDevice(_soa.ry, _h_ry);
            copyToDevice(_soa.rz, _h_rz);
            copyToDevice(_soa.vx, _h_vx);
            copyToDevice(_soa.vy, _h_vy);
            copyToDevice(_soa.vz, _h_vz);
            copyToDevice(_soa.fx, _h_fx);
            copyToDevice(_soa.fy, _h_fy);
            copyToDevice(_soa.fz, _h_fz);
        }
        _host_modified = false;
    }

    // Copy the device buffers back to the host mirror if the device side is dirty.
    void syncHost() {
        if (!_device_modified) return;
        const uint_type n = _num_particles;
        if (n > 0) {
            copyToHost(_h_cell_idx, _soa.cell_idx);
            copyToHost(_h_id, _soa.id);
            copyToHost(_h_rx, _soa.rx);
            copyToHost(_h_ry, _soa.ry);
            copyToHost(_h_rz, _soa.rz);
            copyToHost(_h_vx, _soa.vx);
            copyToHost(_h_vy, _soa.vy);
            copyToHost(_h_vz, _soa.vz);
            copyToHost(_h_fx, _soa.fx);
            copyToHost(_h_fy, _soa.fy);
            copyToHost(_h_fz, _soa.fz);
        }
        _device_modified = false;
    }

    // --- Getters ---

    float_type getInteractionRadius() const { return _interaction_radius; }

    const Vec3u& getNumCellsWithHalo() const { return _num_cells_with_halo; }

    uint_type getNumCells() const { return _num_cells; }

    const Vec3& getCellSize() const { return _cell_size; }

    const Vec3& getBoxMinWithHalo() const { return _box_min_with_halo; }

    const CellIndex& getCellIndexing() const { return _cell_indexing; }

    const uint_type* getBinOffsets() const { return _d_bin_offsets; }

    /**
     * Materialise all particles on the host, syncing from the device first.
     *
     * @return A vector of Particle objects representing the current state of the container.
     *
     * @note This operation can be expensive for large numbers of particles
     */
    std::vector<Particle> getParticles() {
        syncHost();
        std::vector<Particle> particles;
        particles.reserve(_num_particles);
        for (uint_type i = 0; i < _num_particles; ++i) {
            Particle p;
            p.id = _h_id[i];
            p.r = {_h_rx[i], _h_ry[i], _h_rz[i]};
            p.v = {_h_vx[i], _h_vy[i], _h_vz[i]};
            p.f = {_h_fx[i], _h_fy[i], _h_fz[i]};
            particles.push_back(p);
        }
        return particles;
    }

  private:
    /**
     * Device-side computation of the cell indices for all particles based on their position.
     * Leaves the host mirror unchanged and marks the device buffers as modified.
     */
    void updateCellIdx() {
        syncDevice();  // Ensure we have the latest data on the device
        if (_num_particles == 0) return;
        cudaUpdateCellIdxKernel<<<cudaNumBlocks(_num_particles), CUDA_THREADS_PER_BLOCK>>>(
            _soa, _num_particles, _box_min_with_halo, _cell_size, _cell_indexing);
        CUDA_CHECK(cudaGetLastError());
        modifyDevice();  // Mark the device data as modified
    }

    /**
     * Device-side counting sort of the particles by their cell index.
     * Produces the `_d_bin_offsets` array and reorders the SoA buffers contiguously by cell.
     * @note This operation is the counterpart of `sortByCellIdx` in `KokkosSoAContainer`
     */
    void sortByCellIdx() {
        syncDevice();  // Ensure we have the latest data on the device
        if (_num_particles == 0) {
            // With no particles every cell is empty: all offsets are zero.
            CUDA_CHECK(cudaMemset(_d_bin_offsets, 0, (_num_cells + 1) * sizeof(uint_type)));
            return;
        }

        // Count the number of particles in each cell and store in _d_bin_offsets (last slot is 0)
        CUDA_CHECK(cudaMemset(_d_bin_offsets, 0, (_num_cells + 1) * sizeof(uint_type)));
        cudaCountCellsKernel<<<cudaNumBlocks(_num_particles), CUDA_THREADS_PER_BLOCK>>>(_soa.cell_idx, _num_particles, _d_bin_offsets);
        CUDA_CHECK(cudaGetLastError());

        /**
         * Exclusive prefix sum using `thrust::exclusive_scan`
         * Produces per-cell start offsets, with the last slot holding the particle count.
         * -> bin_offsets(c)..bin_offsets(c+1) is cell c.
         */
        thrust::device_ptr<uint_type> offsets(_d_bin_offsets);
        thrust::exclusive_scan(offsets, offsets + (_num_cells + 1), offsets);

        // Seed the write cursor with the start offsets, then scatter to build the
        // permutation (new position -> old index).
        CUDA_CHECK(cudaMemcpy(_d_bin_cursor, _d_bin_offsets, _num_cells * sizeof(uint_type), cudaMemcpyDeviceToDevice));
        cudaScatterPermKernel<<<cudaNumBlocks(_num_particles), CUDA_THREADS_PER_BLOCK>>>(_soa.cell_idx, _num_particles, _d_bin_cursor, _d_perm);
        CUDA_CHECK(cudaGetLastError());

        // Apply the permutation out-of-place into the temporary soa, then swap it in.
        cudaApplyPermutationSoAKernel<<<cudaNumBlocks(_num_particles), CUDA_THREADS_PER_BLOCK>>>(_reorder_tmp_soa, _soa, _d_perm, _num_particles);
        CUDA_CHECK(cudaGetLastError());
        std::swap(_soa, _reorder_tmp_soa);

        CUDA_CHECK(cudaDeviceSynchronize());
        modifyDevice();  // Mark the device data as modified
    }

    template <typename T>
    void copyToDevice(T* dst, const std::vector<T>& src) {
        CUDA_CHECK(cudaMemcpy(dst, src.data(), _num_particles * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void copyToHost(std::vector<T>& dst, const T* src) {
        CUDA_CHECK(cudaMemcpy(dst.data(), src, _num_particles * sizeof(T), cudaMemcpyDeviceToHost));
    }

    static void allocSoA(CudaSoA& soa, uint_type capacity) {
        CUDA_CHECK(cudaMalloc(&soa.cell_idx, capacity * sizeof(uint_type)));
        CUDA_CHECK(cudaMalloc(&soa.id, capacity * sizeof(uint_type)));
        CUDA_CHECK(cudaMalloc(&soa.rx, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.ry, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.rz, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.vx, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.vy, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.vz, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.fx, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.fy, capacity * sizeof(float_type)));
        CUDA_CHECK(cudaMalloc(&soa.fz, capacity * sizeof(float_type)));
    }

    static void freeSoA(CudaSoA& soa) {
        cudaFree(soa.cell_idx);
        cudaFree(soa.id);
        cudaFree(soa.rx);
        cudaFree(soa.ry);
        cudaFree(soa.rz);
        cudaFree(soa.vx);
        cudaFree(soa.vy);
        cudaFree(soa.vz);
        cudaFree(soa.fx);
        cudaFree(soa.fy);
        cudaFree(soa.fz);
        soa = CudaSoA{};
    }

    /**
     * Grow the device buffers to hold at least the necessary number of particles.
     * Doubles the capacity for amortised O(1) growth. Existing device contents are NOT preserved.
     * Growth only happens during the host-side add phase and the next syncDevice refreshes them from the host mirror.
     * @param needed The number of particles that need to be accommodated.
     */
    void ensureDeviceCapacity(uint_type needed) {
        if (needed <= _dev_capacity) return;
        uint_type new_capacity = _dev_capacity == 0 ? 8 : _dev_capacity;
        while (new_capacity < needed) new_capacity *= 2;

        freeSoA(_soa);
        freeSoA(_reorder_tmp_soa);
        cudaFree(_d_perm);
        _d_perm = nullptr;

        allocSoA(_soa, new_capacity);
        allocSoA(_reorder_tmp_soa, new_capacity);
        CUDA_CHECK(cudaMalloc(&_d_perm, new_capacity * sizeof(uint_type)));
        _dev_capacity = new_capacity;
    }

    /**
     * Range checked computation of the cell index for a given position.
     * @param position The position of the particle.
     * @return The cell index corresponding to the position.
     * @throws std::out_of_range if the position is outside the simulation box (with halo).
     * @note This function is used during host-side particle addition to compute the cell index
     */
    uint_type getCellIndex(const Vec3& position) const {
        Vec3u cell_coords;
        for (int d = 0; d < 3; ++d) {
            if (position[d] < _box_min[d] || position[d] >= _box_max[d]) {
                throw std::out_of_range("Position is outside the simulation domain");
            }
            cell_coords[d] = static_cast<uint_type>((position[d] - _box_min_with_halo[d]) / _cell_size[d]);
        }
        return _cell_indexing.encode(cell_coords);
    }
};
