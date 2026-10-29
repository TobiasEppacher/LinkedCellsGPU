#pragma once

#include <sycl/sycl.hpp>

#include <stdexcept>
#include <vector>

#include "ParticleContainer.hpp"
#include "SyclContainerCommon.hpp"
#include "../particle/Particle.hpp"
#include "../traversal/CellIndexing.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Array-of-structs counterpart of SyclSoAContainer: same grid geometry, same host/device
 * dirty-flag sync and the same counting sort (count, hierarchical exclusive scan, scatter,
 * reorder), but the particle state lives in one USM buffer of Particle structs instead of
 * eleven parallel scalar arrays.
 */

/**
 * Raw USM device pointers describing the AoS particle buffer, plus the cell index as a
 * side array -- the index is sort bookkeeping, not particle state, so Particle stays
 * exactly the struct the host containers use. Captured by value into kernels; the
 * container itself is a host object and must never be dereferenced on the device.
 */
struct SyclAoS {
    uint_type* cell_idx = nullptr;
    Particle* particles = nullptr;
};

// --- Container class ---

class SyclAoSContainer : ParticleContainer {
    friend class SyclAoSTraversal;

    // In-order queue so queue.wait() has the same meaning as cudaDeviceSynchronize().
    sycl::queue _queue;

    // Per-level scratch for the hierarchical exclusive scan, one pair per level.
    std::vector<uint_type*> _scan_sums;
    std::vector<uint_type*> _scan_scanned;

    // Host mirror (array-of-structs); always sized to _num_particles.
    std::vector<uint_type> _h_cell_idx;
    std::vector<Particle> _h_particles;

    // Live device particle buffer
    SyclAoS _aos;

    // Device buffer for out-of-place reordering (temporary)
    SyclAoS _reorder_tmp_aos;

    // Permutation array for reordering
    uint_type* _d_perm = nullptr;

    // Allocated capacity of _aos and _reorder_tmp_aos in particles
    uint_type _dev_capacity = 0;

    // Per-cell particle counts, then per-cell start offsets after the scan.
    uint_type* _d_bin_offsets = nullptr;
    uint_type* _d_bin_cursor = nullptr;

    // Per-cell particle counts, used for the exclusive scan.
    uint_type* _d_bin_counts = nullptr;

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
    SyclAoSContainer(Vec3 box_min, Vec3 box_max, float_type interaction_radius)
        : _queue(sycl::property::queue::in_order{}),
          _box_min(box_min),
          _box_max(box_max),
          _interaction_radius(interaction_radius),
          _num_particles(0) {
        // The histogram kernel needs a device atomic matching uint_type.
        if (sizeof(uint_type) == 8 && !_queue.get_device().has(sycl::aspect::atomic64)) {
            throw std::runtime_error(
                "SYCL device lacks aspect::atomic64, which is required by the 64-bit build; "
                "reconfigure and rebuild with -DUSE_64_BIT=Off");
        }

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

        _d_bin_counts = sycl::malloc_device<uint_type>(_num_cells + 1, _queue);
        _d_bin_offsets = sycl::malloc_device<uint_type>(_num_cells + 1, _queue);
        _d_bin_cursor = sycl::malloc_device<uint_type>(_num_cells, _queue);

        // Allocated up front so rebuild() never allocates.
        for (uint_type len = _num_cells + 1; len > 1;) {
            const uint_type groups = numScanGroups(len);
            _scan_sums.push_back(sycl::malloc_device<uint_type>(groups, _queue));
            _scan_scanned.push_back(sycl::malloc_device<uint_type>(groups, _queue));
            len = groups;
        }
    }

    // Raw device ownership makes copying unsafe.
    SyclAoSContainer(const SyclAoSContainer&) = delete;
    SyclAoSContainer& operator=(const SyclAoSContainer&) = delete;

    ~SyclAoSContainer() override {
        freeAoS(_aos);
        freeAoS(_reorder_tmp_aos);
        sycl::free(_d_perm, _queue);
        sycl::free(_d_bin_counts, _queue);
        sycl::free(_d_bin_offsets, _queue);
        sycl::free(_d_bin_cursor, _queue);
        for (uint_type* p : _scan_sums) sycl::free(p, _queue);
        for (uint_type* p : _scan_scanned) sycl::free(p, _queue);
    }

    // Host-side addition of a particle
    void addParticle(const Particle& p) override {
        _h_cell_idx.push_back(getCellIndex(p.r));
        _h_particles.push_back(p);
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
        if (_num_particles > 0) {
            copyToDevice(_aos.cell_idx, _h_cell_idx);
            copyToDevice(_aos.particles, _h_particles);
            _queue.wait();
        }
        _host_modified = false;
    }

    // Copy the device buffers back to the host mirror if the device side is dirty.
    void syncHost() {
        if (!_device_modified) return;
        if (_num_particles > 0) {
            copyToHost(_h_cell_idx, _aos.cell_idx);
            copyToHost(_h_particles, _aos.particles);
            _queue.wait();
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
        return _h_particles;
    }

   private:
    // Recomputes every particle's cell index from its position, on the device.
    void updateCellIdx() {
        syncDevice();
        if (_num_particles == 0) return;

        const SyclAoS p = _aos;
        const uint_type n = _num_particles;
        const Vec3 box_min_with_halo = _box_min_with_halo;
        const Vec3 cell_size = _cell_size;
        const CellIndex cell_indexing = _cell_indexing;

        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)}, sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                if (i >= n) return;
                const Vec3 r = p.particles[i].r;
                const uint_type cx = static_cast<uint_type>((r[0] - box_min_with_halo[0]) / cell_size[0]);
                const uint_type cy = static_cast<uint_type>((r[1] - box_min_with_halo[1]) / cell_size[1]);
                const uint_type cz = static_cast<uint_type>((r[2] - box_min_with_halo[2]) / cell_size[2]);
                p.cell_idx[i] = cell_indexing.encode({cx, cy, cz});
            });
        _queue.wait();
        modifyDevice();
    }

    /**
     * Device-side counting sort of the particles by their cell index.
     * Produces the `_d_bin_offsets` array and reorders the particle buffer contiguously by cell.
     * @note Same pipeline as `SyclSoAContainer::sortByCellIdx`; the reorder kernel moves
     *       whole Particle structs instead of eleven scalars.
     */
    void sortByCellIdx() {
        syncDevice();
        if (_num_particles == 0) {
            _queue.fill(_d_bin_offsets, uint_type(0), _num_cells + 1).wait();
            return;
        }

        const SyclAoS src = _aos;
        const uint_type n = _num_particles;
        const uint_type num_cells = _num_cells;
        uint_type* counts = _d_bin_counts;
        uint_type* offsets = _d_bin_offsets;
        uint_type* cursor = _d_bin_cursor;
        uint_type* perm = _d_perm;

        // Count particles per cell.
        _queue.fill(counts, uint_type(0), num_cells + 1).wait();
        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)}, sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                if (i >= n) return;
                SyclUintAtomicRef{counts[src.cell_idx[i]]}.fetch_add(uint_type(1));
            });
        _queue.wait();

        // Exclusive prefix sum -> start offsets; cell c is [offsets(c), offsets(c+1)).
        exclusiveScan(counts, offsets, num_cells + 1, 0);

        // Scatter into the permutation (new position -> old index).
        _queue.memcpy(cursor, offsets, num_cells * sizeof(uint_type)).wait();
        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)}, sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type old_i = static_cast<uint_type>(item.get_global_id(0));
                if (old_i >= n) return;
                const uint_type new_i = SyclUintAtomicRef{cursor[src.cell_idx[old_i]]}.fetch_add(uint_type(1));
                perm[new_i] = old_i;
            });
        _queue.wait();

        applyPermutation();
        modifyDevice();
    }

    /**
     * Apply `_d_perm` out-of-place, then swap it in.
     * @note Its own function so profilers can tell the reorder kernel apart by name.
     */
    void applyPermutation() {
        const SyclAoS src = _aos;
        const SyclAoS dst = _reorder_tmp_aos;
        const uint_type n = _num_particles;
        const uint_type* perm = _d_perm;
        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)}, sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type new_i = static_cast<uint_type>(item.get_global_id(0));
                if (new_i >= n) return;
                const uint_type old_i = perm[new_i];
                dst.cell_idx[new_i] = src.cell_idx[old_i];
                dst.particles[new_i] = src.particles[old_i];
            });
        _queue.wait();

        std::swap(_aos, _reorder_tmp_aos);
    }

    /**
     * Exclusive prefix sum: each work group scans its chunk, the per-group totals are
     * scanned recursively, and the result is added back as a per-group offset.
     *
     * Hand-written rather than acpp::algorithms::exclusive_scan, whose lookback protocol
     * emits a seq_cst 32-bit atomic store that clang's NVPTX backend cannot select in
     * the cuda:sm_86 flow.
     */
    void exclusiveScan(const uint_type* in, uint_type* out, uint_type n, std::size_t level) {
        if (n == 0) return;
        const uint_type num_groups = numScanGroups(n);
        uint_type* sums = _scan_sums[level];

        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(num_groups) * SYCL_WORK_GROUP_SIZE},
                              sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                const uint_type v = (i < n) ? in[i] : uint_type(0);
                const uint_type excl = sycl::exclusive_scan_over_group(item.get_group(), v, sycl::plus<uint_type>());
                if (i < n) out[i] = excl;
                // Last item of the group holds the inclusive total for the group.
                if (item.get_local_id(0) == SYCL_WORK_GROUP_SIZE - 1) {
                    sums[item.get_group(0)] = excl + v;
                }
            });
        _queue.wait();

        // A single group already produced the complete scan: its offset is zero.
        if (num_groups <= 1) return;

        uint_type* scanned = _scan_scanned[level];
        exclusiveScan(sums, scanned, num_groups, level + 1);

        _queue.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(num_groups) * SYCL_WORK_GROUP_SIZE},
                              sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> item) {
                const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                if (i >= n) return;
                out[i] += scanned[item.get_group(0)];
            });
        _queue.wait();
    }

    template <typename T>
    void copyToDevice(T* dst, const std::vector<T>& src) {
        _queue.memcpy(dst, src.data(), _num_particles * sizeof(T));
    }

    template <typename T>
    void copyToHost(std::vector<T>& dst, const T* src) {
        _queue.memcpy(dst.data(), src, _num_particles * sizeof(T));
    }

    void allocAoS(SyclAoS& aos, uint_type capacity) {
        aos.cell_idx = sycl::malloc_device<uint_type>(capacity, _queue);
        aos.particles = sycl::malloc_device<Particle>(capacity, _queue);
    }

    void freeAoS(SyclAoS& aos) {
        sycl::free(aos.cell_idx, _queue);
        sycl::free(aos.particles, _queue);
        aos = SyclAoS{};
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

        freeAoS(_aos);
        freeAoS(_reorder_tmp_aos);
        sycl::free(_d_perm, _queue);
        _d_perm = nullptr;

        allocAoS(_aos, new_capacity);
        allocAoS(_reorder_tmp_aos, new_capacity);
        _d_perm = sycl::malloc_device<uint_type>(new_capacity, _queue);
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
