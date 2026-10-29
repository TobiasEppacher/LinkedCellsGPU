#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ParticleContainer.hpp"
#include "../particle/Particle.hpp"
#include "../traversal/CellIndexing.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Array-of-structs counterpart of KokkosSoAContainer: same grid geometry, same host/device
 * dirty-flag sync and the same counting sort (count, exclusive scan, scatter, fused
 * reorder), but the particle state lives in one View of Particle structs instead of eleven
 * parallel scalar arrays.
 */

/**
 * Array-of-structs particle buffer in one memory space, plus the cell index as a side
 * array -- the index is sort bookkeeping, not particle state, so Particle stays exactly
 * the struct the host containers use. Views are handles, so the struct is captured by
 * value into kernels; the container itself is a host object and must never be
 * dereferenced on the device.
 */
template <typename Space>
struct KokkosAoS {
    Kokkos::View<uint_type*, Space> cell_idx;
    Kokkos::View<Particle*, Space> particles;

    static KokkosAoS allocate(const std::string& tag, uint_type capacity) {
        return KokkosAoS{
            Kokkos::View<uint_type*, Space>(Kokkos::view_alloc(tag + "cell_idx", Kokkos::WithoutInitializing), capacity),
            Kokkos::View<Particle*, Space>(Kokkos::view_alloc(tag + "particles", Kokkos::WithoutInitializing), capacity)};
    }
};

// --- Container class ---

class KokkosAoSContainer : ParticleContainer {
    friend class KokkosAoSTraversal;
    friend class KokkosC01AoSTraversal;
    friend class KokkosC01ScratchAoSTraversal;

   protected:
    using DeviceAoS = KokkosAoS<Device>;
    using HostAoS = KokkosAoS<Kokkos::HostSpace>;
    using UintView = Kokkos::View<uint_type*, Device>;

    // Host mirror (array-of-structs); valid up to _num_particles.
    HostAoS _host;

    // Live device particle buffer
    DeviceAoS _aos;

    // Device buffer for out-of-place reordering (temporary)
    DeviceAoS _reorder_tmp_aos;

    // Permutation array for reordering
    UintView _perm;

    // Allocated capacity of the host mirror in particles
    uint_type _host_capacity = 0;

    // Allocated capacity of _aos, _reorder_tmp_aos and _perm in particles
    uint_type _dev_capacity = 0;

    // Per-cell start offsets into the particle array
    UintView _bin_offsets;

    // Per-cell write cursor for scattering into the particle array (temporary)
    UintView _bin_cursor;

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
    KokkosAoSContainer(Vec3 box_min, Vec3 box_max, float_type interaction_radius)
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
        _box_min_with_halo = {_box_min[0] - _cell_size[0], _box_min[1] - _cell_size[1], _box_min[2] - _cell_size[2]};
        _box_max_with_halo = {_box_max[0] + _cell_size[0], _box_max[1] + _cell_size[1], _box_max[2] + _cell_size[2]};

        _num_cells = _num_cells_with_halo[0] * _num_cells_with_halo[1] * _num_cells_with_halo[2];
        _cell_indexing = CellIndex(_num_cells_with_halo);

        _host = HostAoS::allocate("host_aos_", 0);
        _bin_offsets = UintView("bin_offsets", _num_cells + 1);
        _bin_cursor = UintView("bin_cursor", _num_cells);
    }

    // Host-side addition of a particle
    void addParticle(const Particle& p) override {
        const uint_type cell_idx = getCellIndex(p.r);  // Throws before any state changes
        ensureHostCapacity(_num_particles + 1);
        const uint_type idx = _num_particles++;
        _host.cell_idx(idx) = cell_idx;
        _host.particles(idx) = p;

        modifyHost();
    }

    uint_type size() const override {
        return _num_particles;
    }

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
        copyActiveRange(_aos, _host);
        _host_modified = false;
    }

    // Copy the device buffers back to the host mirror if the device side is dirty.
    void syncHost() {
        if (!_device_modified) return;
        copyActiveRange(_host, _aos);
        _device_modified = false;
    }

    // --- Getters ---

    float_type getInteractionRadius() const {
        return _interaction_radius;
    }

    const Vec3u& getNumCellsWithHalo() const {
        return _num_cells_with_halo;
    }

    uint_type getNumCells() const {
        return _num_cells;
    }

    const Vec3& getCellSize() const {
        return _cell_size;
    }

    const Vec3& getBoxMinWithHalo() const {
        return _box_min_with_halo;
    }

    const CellIndex& getCellIndexing() const {
        return _cell_indexing;
    }

    // Materialise all particles on the host, syncing from the device first.
    std::vector<Particle> getParticles() {
        syncHost();
        const Particle* begin = _host.particles.data();
        return std::vector<Particle>(begin, begin + _num_particles);
    }

    // bin_offsets has _num_cells+1 entries; particles in cell c are stored
    // contiguously at indices [bin_offsets(c), bin_offsets(c+1)) after a rebuild.
    UintView getBinOffsets() const {
        return _bin_offsets;
    }

  protected:
    /**
     * Device-side computation of the cell indices for all particles based on their position.
     * Leaves the host mirror unchanged and marks the device buffers as modified.
     */
    void updateCellIdx() {
        syncDevice();  // Ensure we have the latest data on the device
        if (_num_particles == 0) return;

        const DeviceAoS aos = _aos;
        const Vec3 box_min_with_halo = _box_min_with_halo;
        const Vec3 cell_size = _cell_size;
        const CellIndex cell_indexing = _cell_indexing;

        Kokkos::parallel_for("UpdateCellIdx", Kokkos::RangePolicy<Device>(0, _num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                const Vec3 r = aos.particles(i).r;
                const uint_type cx = static_cast<uint_type>((r[0] - box_min_with_halo[0]) / cell_size[0]);
                const uint_type cy = static_cast<uint_type>((r[1] - box_min_with_halo[1]) / cell_size[1]);
                const uint_type cz = static_cast<uint_type>((r[2] - box_min_with_halo[2]) / cell_size[2]);
                aos.cell_idx(i) = cell_indexing.encode({cx, cy, cz});
            });
        modifyDevice();  // Mark the device data as modified
    }

    /**
     * Device-side counting sort of the particles by their cell index.
     * Produces the `_bin_offsets` array and reorders the particle buffer contiguously by cell.
     * @note Same pipeline as `KokkosSoAContainer::sortByCellIdx`; the reorder kernel moves
     *       whole Particle structs instead of eleven scalars.
     */
    void sortByCellIdx() {
        syncDevice();  // Ensure we have the latest data on the device
        if (_num_particles == 0) {
            // With no particles every cell is empty: all offsets are zero.
            Kokkos::deep_copy(_bin_offsets, uint_type(0));
            return;
        }

        const uint_type num_particles = _num_particles;
        const uint_type num_cells = _num_cells;
        const UintView offsets = _bin_offsets;
        const UintView cursor = _bin_cursor;
        const UintView perm = _perm;
        const DeviceAoS src = _aos;

        // Count the number of particles in each cell and store in _bin_offsets (last slot is 0)
        Kokkos::deep_copy(offsets, uint_type(0));
        Kokkos::parallel_for("CountCells", Kokkos::RangePolicy<Device>(0, num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                Kokkos::atomic_fetch_add(&offsets(src.cell_idx(i)), uint_type(1));
            });

        /**
         * Exclusive prefix sum in place, producing per-cell start offsets, with the last
         * slot holding the particle count.
         * -> bin_offsets(c)..bin_offsets(c+1) is cell c.
         */
        Kokkos::parallel_scan("ScanCells", Kokkos::RangePolicy<Device>(0, num_cells + 1),
            KOKKOS_LAMBDA(uint_type c, uint_type& running, const bool final) {
                const uint_type count = offsets(c);
                if (final) offsets(c) = running;
                running += count;
            });

        // Seed the write cursor with the start offsets, then scatter to build the
        // permutation (new position -> old index).
        Kokkos::deep_copy(cursor, Kokkos::subview(offsets, Kokkos::make_pair(uint_type(0), num_cells)));
        Kokkos::parallel_for("ScatterPerm", Kokkos::RangePolicy<Device>(0, num_particles),
            KOKKOS_LAMBDA(uint_type old_i) {
                const uint_type new_i = Kokkos::atomic_fetch_add(&cursor(src.cell_idx(old_i)), uint_type(1));
                perm(new_i) = old_i;
            });

        applyPermutation();

        Kokkos::fence();
        modifyDevice();  // Mark the device data as modified
    }

    /**
     * Apply `_perm` out-of-place into the temporary buffer, then swap it in.
     * @note Its own function so profilers can tell the reorder kernel apart by name.
     */
    void applyPermutation() {
        const UintView perm = _perm;
        const DeviceAoS src = _aos;
        const DeviceAoS dst = _reorder_tmp_aos;
        Kokkos::parallel_for("Reorder", Kokkos::RangePolicy<Device>(0, _num_particles),
            KOKKOS_LAMBDA(uint_type new_i) {
                const uint_type old_i = perm(new_i);
                dst.cell_idx(new_i) = src.cell_idx(old_i);
                dst.particles(new_i) = src.particles(old_i);
            });
        std::swap(_aos, _reorder_tmp_aos);
    }

    /**
     * Copy the [0, _num_particles) range of both arrays from one AoS to the other.
     * The two AoSs may live in different memory spaces; the slices are contiguous.
     */
    template <typename DstAoS, typename SrcAoS>
    void copyActiveRange(DstAoS& dst, SrcAoS& src) const {
        if (_num_particles == 0) return;
        const auto range = Kokkos::make_pair(uint_type(0), _num_particles);
        Kokkos::deep_copy(Kokkos::subview(dst.cell_idx, range), Kokkos::subview(src.cell_idx, range));
        Kokkos::deep_copy(Kokkos::subview(dst.particles, range), Kokkos::subview(src.particles, range));
    }

    /**
     * Grow the host mirror to hold at least the necessary number of particles, preserving
     * the particles added so far. Doubles the capacity for amortised O(1) growth.
     * @param needed The number of particles that need to be accommodated.
     */
    void ensureHostCapacity(uint_type needed) {
        if (needed <= _host_capacity) return;
        uint_type new_capacity = _host_capacity == 0 ? 8 : _host_capacity;
        while (new_capacity < needed) new_capacity *= 2;
        Kokkos::resize(_host.cell_idx, new_capacity);
        Kokkos::resize(_host.particles, new_capacity);
        _host_capacity = new_capacity;
    }

    /**
     * Grow the device buffers to hold at least the necessary number of particles.
     * Doubles the capacity for amortised O(1) growth. Existing device contents are NOT preserved;
     * the caller is syncDevice(), which refreshes them from the host mirror right after.
     * @param needed The number of particles that need to be accommodated.
     */
    void ensureDeviceCapacity(uint_type needed) {
        if (needed <= _dev_capacity) return;
        uint_type new_capacity = _dev_capacity == 0 ? 8 : _dev_capacity;
        while (new_capacity < needed) new_capacity *= 2;

        // Release before reallocating so the peak footprint stays at one set of buffers.
        _aos = DeviceAoS{};
        _reorder_tmp_aos = DeviceAoS{};
        _perm = UintView{};
        _aos = DeviceAoS::allocate("aos_", new_capacity);
        _reorder_tmp_aos = DeviceAoS::allocate("reorder_tmp_aos_", new_capacity);
        _perm = UintView(Kokkos::view_alloc("perm", Kokkos::WithoutInitializing), new_capacity);
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
