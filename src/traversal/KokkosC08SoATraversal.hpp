#pragma once

#include "../container/KokkosSoAContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos C08 traversal on a struct-of-arrays (SoA) container.
 */
class KokkosC08SoATraversal {
    KokkosSoAContainer* _container;

    public:
    KokkosC08SoATraversal(KokkosSoAContainer* container)
        : _container(container) {}

    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        auto ids = _container->_soa.id;
        auto rx = _container->_soa.rx;
        auto ry = _container->_soa.ry;
        auto rz = _container->_soa.rz;
        auto vx = _container->_soa.vx;
        auto vy = _container->_soa.vy;
        auto vz = _container->_soa.vz;
        auto fx = _container->_soa.fx;
        auto fy = _container->_soa.fy;
        auto fz = _container->_soa.fz;

        Kokkos::parallel_for(
            "forEachParticle", Kokkos::RangePolicy<Device>(0, _container->size()),
            KOKKOS_LAMBDA(uint_type i) {
                Particle particle_i;
                particle_i.id = ids(i);
                particle_i.r = {rx(i), ry(i), rz(i)};
                particle_i.v = {vx(i), vy(i), vz(i)};
                particle_i.f = {fx(i), fy(i), fz(i)};
                func(particle_i);
                ids(i) = particle_i.id;
                rx(i) = particle_i.r[0];
                ry(i) = particle_i.r[1];
                rz(i) = particle_i.r[2];
                vx(i) = particle_i.v[0];
                vy(i) = particle_i.v[1];
                vz(i) = particle_i.v[2];
                fx(i) = particle_i.f[0];
                fy(i) = particle_i.f[1];
                fz(i) = particle_i.f[2];
            });
        Kokkos::fence();
        _container->modifyDevice();
    }

    // Requires an up-to-date cell binning: call the container's rebuild() first.
    template <typename Func>
    void applyPairwise(Func func) {
        const auto bin_offsets = _container->getBinOffsets();
        const Vec3u num_cells_per_dim = _container->getNumCellsWithHalo();
        const float_type interaction_radius_sq = _container->getInteractionRadius() * _container->getInteractionRadius();

        // Extract device views for kernel to capture.
        // Dereference of _container is invalid on device
        auto ids = _container->_soa.id;
        auto rx = _container->_soa.rx;
        auto ry = _container->_soa.ry;
        auto rz = _container->_soa.rz;
        auto vx = _container->_soa.vx;
        auto vy = _container->_soa.vy;
        auto vz = _container->_soa.vz;
        auto fx = _container->_soa.fx;
        auto fy = _container->_soa.fy;
        auto fz = _container->_soa.fz;
        const CellIndex cell_indexing = _container->_cell_indexing;

        // Inner cells act as base cells
        // We ignore lower halo, as we only use reflecting boundaries
        const uint_type cells_x = num_cells_per_dim[0] - 2;
        const uint_type cells_y = num_cells_per_dim[1] - 2;
        const uint_type cells_z = num_cells_per_dim[2] - 2;

        using team_policy_t = Kokkos::TeamPolicy<Device>;
        using member_t = team_policy_t::member_type;

        for (uint_type color = 0; color < 8; color++) {
            // Per-dimension parity of this color's base cells
            const uint_type offset_x = color & 1;
            const uint_type offset_y = (color >> 1) & 1;
            const uint_type offset_z = (color >> 2) & 1;

            // Requires 2 or more cells in each dimension, else division by 0 in parallel_for
            const uint_type color_cells_x = (cells_x - offset_x + 1) / 2;
            const uint_type color_cells_y = (cells_y - offset_y + 1) / 2;
            const uint_type color_cells_z = (cells_z - offset_z + 1) / 2;
            const uint_type league_size = color_cells_x * color_cells_y * color_cells_z;

            // One thread team per base cell of this color
            team_policy_t policy(static_cast<int>(league_size), Kokkos::AUTO());

            Kokkos::parallel_for(
                "lc_c08_force", policy,
                KOKKOS_LAMBDA(const member_t& team) {
                    const uint_type league_rank = static_cast<uint_type>(team.league_rank());
                    const uint_type base_x = 1 + offset_x + 2 * (league_rank % color_cells_x);
                    const uint_type base_y = 1 + offset_y + 2 * ((league_rank / color_cells_x) % color_cells_y);
                    const uint_type base_z = 1 + offset_z + 2 * (league_rank / (color_cells_x * color_cells_y));

                    // Particles of block cell k (= dx + 2*dy + 4*dz) are numbered [block_offsets[k], block_offsets[k + 1]) within the block
                    uint_type block_offsets[9];
                    block_offsets[0] = 0;
                    // Global particle range [global_cell_begin[k], global_cell_end[k]) of each block cell
                    uint_type global_cell_begin[8];
                    uint_type global_cell_end[8];

                    for (uint_type delta_z = 0; delta_z <= 1; ++delta_z)
                    for (uint_type delta_y = 0; delta_y <= 1; ++delta_y)
                    for (uint_type delta_x = 0; delta_x <= 1; ++delta_x) {
                        const uint_type k = delta_x + 2 * delta_y + 4 * delta_z;
                        const uint_type cell_idx = cell_indexing.encode({base_x + delta_x, base_y + delta_y, base_z + delta_z});
                        global_cell_begin[k] = bin_offsets(cell_idx);
                        global_cell_end[k] = bin_offsets(cell_idx + 1);
                        block_offsets[k + 1] = block_offsets[k] + global_cell_end[k] - global_cell_begin[k];
                    }

                    // Interacting cell pairs of the block: base cell with itself and the 13 half-neighborhood pairs
                    constexpr uint_type num_cell_pairs = 14;
                    const uint_type cell_pairs[num_cell_pairs][2] = {
                        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                        {0, 7}, {1, 2}, {1, 4}, {1, 6}, {2, 4}, {2, 5}, {3, 4}
                    };

                    // One thread per block particle, accumulating forces from all pairs involving its cell
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, block_offsets[8]), [&](uint_type block_i) {
                        uint_type own_cell = 0;
                        while (block_i >= block_offsets[own_cell + 1]) ++own_cell;
                        const uint_type i = global_cell_begin[own_cell] + (block_i - block_offsets[own_cell]);

                        Particle particle_i;
                        particle_i.id = ids(i);
                        particle_i.r = {rx(i), ry(i), rz(i)};
                        particle_i.v = {vx(i), vy(i), vz(i)};
                        particle_i.f = {fx(i), fy(i), fz(i)};

                        Vec3 force = {float_type(0), float_type(0), float_type(0)};
                        for (uint_type pair = 0; pair < num_cell_pairs; ++pair) {
                            uint_type other_cell;
                            if (cell_pairs[pair][0] == own_cell) other_cell = cell_pairs[pair][1];
                            else if (cell_pairs[pair][1] == own_cell) other_cell = cell_pairs[pair][0];
                            else continue;

                            for (uint_type j = global_cell_begin[other_cell]; j < global_cell_end[other_cell]; ++j) {
                                if (j == i) continue;
                                Particle particle_j;
                                particle_j.id = ids(j);
                                particle_j.r = {rx(j), ry(j), rz(j)};
                                particle_j.v = {vx(j), vy(j), vz(j)};
                                particle_j.f = {fx(j), fy(j), fz(j)};
                                if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                                const Vec3 f = func(particle_i, particle_j);
                                force[0] += f[0];
                                force[1] += f[1];
                                force[2] += f[2];
                            }
                        }

                        fx(i) += force[0];
                        fy(i) += force[1];
                        fz(i) += force[2];
                    });
                });
        }
        Kokkos::fence();
        _container->modifyDevice();
    }
};
