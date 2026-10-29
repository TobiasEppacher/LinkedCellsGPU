#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ParticleContainer.hpp"
#include "../particle/Particle.hpp"
#include "../traversal/CellIndexing.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/KokkosVector.hpp"
#include "../util/TypeDefinitions.hpp"

struct AoSCell {
    KokkosVector<Particle, Host> data;

    explicit AoSCell() : data(0) {}

    // Pre-allocates n slots and marks all n as occupied (for fixed-size scenarios).
    explicit AoSCell(size_t n) : data(n) {}

    size_t size() const { return data.size(); }

    void clear() { data.clear(); }

    void addParticle(const Particle& value) {
        data.push_back(value);
    }

    template <typename Func>
    void forEach(Func func, size_t end = std::numeric_limits<size_t>::max()) {
        size_t n = Kokkos::min(end, data.size());
        for (size_t i = 0; i < n; ++i) {
            func(data(i));
        }
    }

    template <typename Func>
    void forEachKokkos(Func func) {
        Kokkos::parallel_for("forEachKokkos", Kokkos::RangePolicy<Host>(0, data.size()),
                             KOKKOS_CLASS_LAMBDA(int i) { func(data(i)); });
    }
};

class CpuAoSContainer : ParticleContainer {
    friend class CpuTraversal;

    Vec3 _box_min;
    Vec3 _box_max;
    float_type _interaction_radius;

    Vec3u _num_cells_with_halo;
    Vec3 _cell_size;
    Vec3 _box_min_with_halo;
    Vec3 _box_max_with_halo;
    std::vector<AoSCell> _cells;  // flat 1D, indexed via cellIndex()

    CellIndex _cell_indexing;

   public:
    CpuAoSContainer(Vec3 box_min, Vec3 box_max, float_type interaction_radius)
        : _box_min(box_min), _box_max(box_max), _interaction_radius(interaction_radius) {
        _num_cells_with_halo = {static_cast<uint_type>((_box_max[0] - _box_min[0]) / _interaction_radius) + 2,
                             static_cast<uint_type>((_box_max[1] - _box_min[1]) / _interaction_radius) + 2,
                             static_cast<uint_type>((_box_max[2] - _box_min[2]) / _interaction_radius) + 2};
        _cell_size = {(_box_max[0] - _box_min[0]) / (_num_cells_with_halo[0] - 2),
                     (_box_max[1] - _box_min[1]) / (_num_cells_with_halo[1] - 2),
                     (_box_max[2] - _box_min[2]) / (_num_cells_with_halo[2] - 2)};
        _box_min_with_halo = {_box_min[0] - _cell_size[0], _box_min[1] - _cell_size[1], _box_min[2] - _cell_size[2]};
        _box_max_with_halo = {_box_max[0] + _cell_size[0], _box_max[1] + _cell_size[1], _box_max[2] + _cell_size[2]};
        _cells.resize(_num_cells_with_halo[0] * _num_cells_with_halo[1] * _num_cells_with_halo[2]);
        _cell_indexing = CellIndex(_num_cells_with_halo);
    }

    void addParticle(const Particle& value) override {
        auto idx = getCellIndex(value.r);
        _cells[idx].addParticle(value);
    }

    uint_type size() const override {
        uint_type total = 0;
        for (const auto& c : _cells) total += c.size();
        return total;
    }

    void rebuild() override {
        std::vector<Particle> all_particles = getParticles();
        for (auto& c : _cells) c.clear();
        for (const auto& p : all_particles) addParticle(p);
    }

    // --- Getters ---
    float_type getInteractionRadius() const {
        return _interaction_radius;
    }

    const Vec3u& getNumCellsWithHalo() const {
        return _num_cells_with_halo;
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

    std::vector<Particle> getParticles() const {
        std::vector<Particle> particles;
        particles.reserve(size());
        for (const auto& c : _cells) {
            for (uint_type i = 0; i < c.size(); ++i) {
                particles.push_back(c.data(i));
            }
        }
        return particles;
    }

   private:
    AoSCell& getCell(uint_type x, uint_type y, uint_type z) {
        return _cells[_cell_indexing.encode({x, y, z})];
    }

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
