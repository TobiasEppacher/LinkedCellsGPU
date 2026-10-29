#pragma once

#include "../container/KokkosSoAContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/KokkosScratch.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos C08 traversal on a struct-of-arrays (SoA) container.
 */
class KokkosC08ScratchSoATraversal {
    KokkosSoAContainer* _container;
    bool _warned_scratch_fallback = false;

    public:
    KokkosC08ScratchSoATraversal(KokkosSoAContainer* container)
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

        // Precompute maximum number of particles in a 2x2x2 block to get scratch memory size
        uint_type max_block = 0;
        Kokkos::parallel_reduce(
            "lc_c08_max_block",
            Kokkos::MDRangePolicy<Device, Kokkos::Rank<3>>({1, 1, 1}, {num_cells_per_dim[0] - 1, num_cells_per_dim[1] - 1, num_cells_per_dim[2] - 1}),
            KOKKOS_LAMBDA(uint_type cell_x, uint_type cell_y, uint_type cell_z, uint_type& local_max) {
                uint_type count = 0;
                for (uint_type delta_z = 0; delta_z <= 1; ++delta_z)
                for (uint_type delta_y = 0; delta_y <= 1; ++delta_y)
                for (uint_type delta_x = 0; delta_x <= 1; ++delta_x) {
                    const uint_type cell_idx = cell_indexing.encode({cell_x + delta_x, cell_y + delta_y, cell_z + delta_z});
                    count += bin_offsets(cell_idx + 1) - bin_offsets(cell_idx);
                }
                if (count > local_max) local_max = count;
            },
            Kokkos::Max<uint_type>(max_block)
        );

        using team_policy_t = Kokkos::TeamPolicy<Device>;
        using member_t = team_policy_t::member_type;
        using ScratchFloat = Kokkos::View<float_type*, Device::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using ScratchUint = Kokkos::View<uint_type*, Device::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // Byte count for the staged particles: one id array plus nine r/v/f arrays
        const size_t scratch_bytes = ScratchUint::shmem_size(max_block) + 9 * ScratchFloat::shmem_size(max_block);
        const int scratch_level = selectScratchLevel(scratch_bytes, "lc_c08_force", _warned_scratch_fallback);

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
            policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes));

            Kokkos::parallel_for(
                "lc_c08_force", policy,
                KOKKOS_LAMBDA(const member_t& team) {
                    const uint_type league_rank = static_cast<uint_type>(team.league_rank());
                    const uint_type base_x = 1 + offset_x + 2 * (league_rank % color_cells_x);
                    const uint_type base_y = 1 + offset_y + 2 * ((league_rank / color_cells_x) % color_cells_y);
                    const uint_type base_z = 1 + offset_z + 2 * (league_rank / (color_cells_x * color_cells_y));

                    // Allocate scratch memory
                    ScratchUint s_id(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_rx(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_ry(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_rz(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_vx(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_vy(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_vz(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_fx(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_fy(team.team_scratch(scratch_level), max_block);
                    ScratchFloat s_fz(team.team_scratch(scratch_level), max_block);

                    // Staged range of block cell k (= dx + 2*dy + 4*dz) is [block_offsets[k], block_offsets[k + 1])
                    uint_type block_offsets[9];
                    block_offsets[0] = 0;
                    // Global index of the first particle of each block cell, for the force write-back
                    uint_type global_cell_begin[8];

                    for (uint_type delta_z = 0; delta_z <= 1; ++delta_z)
                    for (uint_type delta_y = 0; delta_y <= 1; ++delta_y)
                    for (uint_type delta_x = 0; delta_x <= 1; ++delta_x) {
                        const uint_type k = delta_x + 2 * delta_y + 4 * delta_z;
                        const uint_type cell_idx = cell_indexing.encode({base_x + delta_x, base_y + delta_y, base_z + delta_z});
                        const uint_type src_begin = bin_offsets(cell_idx);
                        const uint_type count = bin_offsets(cell_idx + 1) - src_begin;
                        const uint_type dst = block_offsets[k];
                        global_cell_begin[k] = src_begin;
                        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, count), [&](uint_type n) {
                            const uint_type src = src_begin + n;
                            s_id(dst + n) = ids(src);
                            s_rx(dst + n) = rx(src);
                            s_ry(dst + n) = ry(src);
                            s_rz(dst + n) = rz(src);
                            s_vx(dst + n) = vx(src);
                            s_vy(dst + n) = vy(src);
                            s_vz(dst + n) = vz(src);
                            s_fx(dst + n) = fx(src);
                            s_fy(dst + n) = fy(src);
                            s_fz(dst + n) = fz(src);
                        });
                        block_offsets[k + 1] = dst + count;
                    }

                    team.team_barrier();  // staging complete before the force loop reads scratch

                    // Interacting cell pairs of the block: base cell with itself and the 13 half-neighborhood pairs
                    constexpr uint_type num_cell_pairs = 14;
                    const uint_type cell_pairs[num_cell_pairs][2] = {
                        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                        {0, 7}, {1, 2}, {1, 4}, {1, 6}, {2, 4}, {2, 5}, {3, 4}
                    };

                    // One thread per staged particle, accumulating forces from all pairs involving its cell
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, block_offsets[8]), [&](uint_type i) {
                        uint_type own_cell = 0;
                        while (i >= block_offsets[own_cell + 1]) ++own_cell;

                        Particle particle_i;
                        particle_i.id = s_id(i);
                        particle_i.r = {s_rx(i), s_ry(i), s_rz(i)};
                        particle_i.v = {s_vx(i), s_vy(i), s_vz(i)};
                        particle_i.f = {s_fx(i), s_fy(i), s_fz(i)};

                        Vec3 force = {float_type(0), float_type(0), float_type(0)};
                        for (uint_type pair = 0; pair < num_cell_pairs; ++pair) {
                            uint_type other_cell;
                            if (cell_pairs[pair][0] == own_cell) other_cell = cell_pairs[pair][1];
                            else if (cell_pairs[pair][1] == own_cell) other_cell = cell_pairs[pair][0];
                            else continue;

                            for (uint_type j = block_offsets[other_cell]; j < block_offsets[other_cell + 1]; ++j) {
                                if (j == i) continue;
                                Particle particle_j;
                                particle_j.id = s_id(j);
                                particle_j.r = {s_rx(j), s_ry(j), s_rz(j)};
                                particle_j.v = {s_vx(j), s_vy(j), s_vz(j)};
                                particle_j.f = {s_fx(j), s_fy(j), s_fz(j)};
                                if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                                const Vec3 f = func(particle_i, particle_j);
                                force[0] += f[0];
                                force[1] += f[1];
                                force[2] += f[2];
                            }
                        }

                        const uint_type global_i = global_cell_begin[own_cell] + (i - block_offsets[own_cell]);
                        fx(global_i) += force[0];
                        fy(global_i) += force[1];
                        fz(global_i) += force[2];
                    });
                });
        }
        Kokkos::fence();
        _container->modifyDevice();
    }
};
