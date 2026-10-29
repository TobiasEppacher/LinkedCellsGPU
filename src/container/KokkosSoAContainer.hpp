#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ParticleContainer.hpp"
#include "../traversal/CellIndexing.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Kokkos counterpart of CudaSoAContainer, structured the same way: separate host and
 * device buffers with an explicit dirty flag pair, and a hand-written counting sort
 * (count, exclusive scan, scatter, reorder)
 */

/**
 * Structure-of-arrays particle buffers in one memory space. Views are handles, so the
 * struct is captured by value into kernels; the container itself is a host object and
 * must never be dereferenced on the device. Counterpart of CudaSoA / SyclSoA.
 */
template <typename Space>
struct KokkosSoA {
    Kokkos::View<uint_type*, Space> cell_idx;
    Kokkos::View<uint_type*, Space> id;
    Kokkos::View<float_type*, Space> rx;
    Kokkos::View<float_type*, Space> ry;
    Kokkos::View<float_type*, Space> rz;
    Kokkos::View<float_type*, Space> vx;
    Kokkos::View<float_type*, Space> vy;
    Kokkos::View<float_type*, Space> vz;
    Kokkos::View<float_type*, Space> fx;
    Kokkos::View<float_type*, Space> fy;
    Kokkos::View<float_type*, Space> fz;

    static KokkosSoA allocate(const std::string& tag, uint_type capacity) {
        const auto uints = [&](const char* name) {
            return Kokkos::View<uint_type*, Space>(Kokkos::view_alloc(tag + name, Kokkos::WithoutInitializing), capacity);
        };
        const auto floats = [&](const char* name) {
            return Kokkos::View<float_type*, Space>(Kokkos::view_alloc(tag + name, Kokkos::WithoutInitializing), capacity);
        };
        return KokkosSoA{uints("cell_idx"), uints("id"),  floats("rx"), floats("ry"), floats("rz"), floats("vx"),
                         floats("vy"),      floats("vz"), floats("fx"), floats("fy"), floats("fz")};
    }

    // Visits all arrays in a fixed order, so bulk operations do not repeat the list.
    template <typename Func>
    void forEachArray(Func func) {
        func(cell_idx);
        func(id);
        func(rx);
        func(ry);
        func(rz);
        func(vx);
        func(vy);
        func(vz);
        func(fx);
        func(fy);
        func(fz);
    }

    // Same, over the corresponding arrays of two SoAs (possibly in different spaces).
    template <typename OtherSoA, typename Func>
    void forEachArrayPair(OtherSoA& other, Func func) {
        func(cell_idx, other.cell_idx);
        func(id, other.id);
        func(rx, other.rx);
        func(ry, other.ry);
        func(rz, other.rz);
        func(vx, other.vx);
        func(vy, other.vy);
        func(vz, other.vz);
        func(fx, other.fx);
        func(fy, other.fy);
        func(fz, other.fz);
    }
};

// --- Container class ---

class KokkosSoAContainer : ParticleContainer {
    friend class KokkosSoATraversal;
    friend class KokkosSoANewton3Traversal;
    friend class KokkosC01SoATraversal;
    friend class KokkosC01ScratchSoATraversal;
    friend class KokkosC08SoATraversal;
    friend class KokkosC08ScratchSoATraversal;

   protected:
    using DeviceSoA = KokkosSoA<Device>;
    using HostSoA = KokkosSoA<Kokkos::HostSpace>;
    using UintView = Kokkos::View<uint_type*, Device>;

    // Host mirror (structure-of-arrays); valid up to _num_particles.
    HostSoA _host;

    // Live device particle buffers
    DeviceSoA _soa;

    // Device buffer for out-of-place reordering (temporary)
    DeviceSoA _reorder_tmp_soa;

    // Permutation array for reordering
    UintView _perm;

    // Allocated capacity of the host mirror in particles
    uint_type _host_capacity = 0;

    // Allocated capacity of _soa, _reorder_tmp_soa and _perm in particles
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
    KokkosSoAContainer(Vec3 box_min, Vec3 box_max, float_type interaction_radius)
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

        _host = HostSoA::allocate("host_", 0);
        _bin_offsets = UintView("bin_offsets", _num_cells + 1);
        _bin_cursor = UintView("bin_cursor", _num_cells);
    }

    // Host-side addition of a particle
    void addParticle(const Particle& p) override {
        const uint_type cell_idx = getCellIndex(p.r);  // Throws before any state changes
        ensureHostCapacity(_num_particles + 1);
        const uint_type idx = _num_particles++;
        _host.cell_idx(idx) = cell_idx;
        _host.id(idx) = p.id;
        _host.rx(idx) = p.r[0];
        _host.ry(idx) = p.r[1];
        _host.rz(idx) = p.r[2];
        _host.vx(idx) = p.v[0];
        _host.vy(idx) = p.v[1];
        _host.vz(idx) = p.v[2];
        _host.fx(idx) = p.f[0];
        _host.fy(idx) = p.f[1];
        _host.fz(idx) = p.f[2];

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
        copyActiveRange(_soa, _host);
        _host_modified = false;
    }

    // Copy the device buffers back to the host mirror if the device side is dirty.
    void syncHost() {
        if (!_device_modified) return;
        copyActiveRange(_host, _soa);
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
        std::vector<Particle> particles;
        particles.reserve(_num_particles);
        for (uint_type i = 0; i < _num_particles; ++i) {
            Particle p;
            p.id = _host.id(i);
            p.r = {_host.rx(i), _host.ry(i), _host.rz(i)};
            p.v = {_host.vx(i), _host.vy(i), _host.vz(i)};
            p.f = {_host.fx(i), _host.fy(i), _host.fz(i)};
            particles.push_back(p);
        }
        return particles;
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

        const DeviceSoA soa = _soa;
        const Vec3 box_min_with_halo = _box_min_with_halo;
        const Vec3 cell_size = _cell_size;
        const CellIndex cell_indexing = _cell_indexing;

        Kokkos::parallel_for("UpdateCellIdx", Kokkos::RangePolicy<Device>(0, _num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                const uint_type cx = static_cast<uint_type>((soa.rx(i) - box_min_with_halo[0]) / cell_size[0]);
                const uint_type cy = static_cast<uint_type>((soa.ry(i) - box_min_with_halo[1]) / cell_size[1]);
                const uint_type cz = static_cast<uint_type>((soa.rz(i) - box_min_with_halo[2]) / cell_size[2]);
                soa.cell_idx(i) = cell_indexing.encode({cx, cy, cz});
            });
        modifyDevice();  // Mark the device data as modified
    }

    /**
     * Device-side counting sort of the particles by their cell index.
     * Produces the `_bin_offsets` array and reorders the SoA buffers contiguously by cell.
     * @note This operation is the counterpart of `cudaCountCellsKernel` and friends in `CudaSoAContainer`
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
        const DeviceSoA src = _soa;

        // Count the number of particles in each cell and store in _bin_offsets (last slot is 0)
        Kokkos::deep_copy(offsets, uint_type(0));
        Kokkos::parallel_for("CountCells", Kokkos::RangePolicy<Device>(0, num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                Kokkos::atomic_fetch_add(&offsets(src.cell_idx(i)), uint_type(1));
            });

        /**
         * Exclusive prefix sum in place, the counterpart of `thrust::exclusive_scan`.
         * Produces per-cell start offsets, with the last slot holding the particle count.
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
     * Apply `_perm` out-of-place into the temporary soa, then swap it in.
     * One fused kernel for all arrays, against BinSort's two kernels per array.
     * @note Its own function so profilers can tell the reorder kernel apart by name.
     */
    void applyPermutation() {
        const UintView perm = _perm;
        const DeviceSoA src = _soa;
        const DeviceSoA dst = _reorder_tmp_soa;
        Kokkos::parallel_for("Reorder", Kokkos::RangePolicy<Device>(0, _num_particles),
            KOKKOS_LAMBDA(uint_type new_i) {
                const uint_type old_i = perm(new_i);
                dst.cell_idx(new_i) = src.cell_idx(old_i);
                dst.id(new_i) = src.id(old_i);
                dst.rx(new_i) = src.rx(old_i);
                dst.ry(new_i) = src.ry(old_i);
                dst.rz(new_i) = src.rz(old_i);
                dst.vx(new_i) = src.vx(old_i);
                dst.vy(new_i) = src.vy(old_i);
                dst.vz(new_i) = src.vz(old_i);
                dst.fx(new_i) = src.fx(old_i);
                dst.fy(new_i) = src.fy(old_i);
                dst.fz(new_i) = src.fz(old_i);
            });
        std::swap(_soa, _reorder_tmp_soa);
    }

    /**
     * Copy the [0, _num_particles) range of every array from one SoA to the other.
     * The two SoAs may live in different memory spaces; the slices are contiguous.
     */
    template <typename DstSoA, typename SrcSoA>
    void copyActiveRange(DstSoA& dst, SrcSoA& src) const {
        if (_num_particles == 0) return;
        const auto range = Kokkos::make_pair(uint_type(0), _num_particles);
        dst.forEachArrayPair(src, [&](auto& dst_view, auto& src_view) {
            Kokkos::deep_copy(Kokkos::subview(dst_view, range), Kokkos::subview(src_view, range));
        });
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
        _host.forEachArray([&](auto& view) { Kokkos::resize(view, new_capacity); });
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
        _soa = DeviceSoA{};
        _reorder_tmp_soa = DeviceSoA{};
        _perm = UintView{};
        _soa = DeviceSoA::allocate("soa_", new_capacity);
        _reorder_tmp_soa = DeviceSoA::allocate("reorder_tmp_", new_capacity);
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
