#pragma once

#include <algorithm>

#include "../container/KokkosSoAContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/KokkosScratch.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos cell-grouped particle-based traversal on a struct-of-arrays (SoA) container.
 * The grouping of threads-per-particle into cells is done via a Kokkos::TeamPolicy
 * for each inner cell, as in `KokkosC01SoATraversal`.
 * The difference is that the particles of the neighborhood are staged into
 * a shared team scratch memory before the force loop, so particle reads are then
 * from the shared scratch instead of global memory.
 */
class KokkosC01ScratchSoATraversal {
    KokkosSoAContainer* _container;
    bool _warned_scratch_fallback = false;

    public:
    KokkosC01ScratchSoATraversal(KokkosSoAContainer* container)
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
        const uint_type num_cells = _container->getNumCells();
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

        // One thread team per inner cell
        const uint_type cells_x = num_cells_per_dim[0] - 2;
        const uint_type cells_y = num_cells_per_dim[1] - 2;
        const uint_type cells_z = num_cells_per_dim[2] - 2;
        const uint_type league_size = cells_x * cells_y * cells_z;

        // Max particles per cell
        uint_type max_particles_per_cell = 0;
        Kokkos::parallel_reduce(
            "lc_cell_max_particles", Kokkos::RangePolicy<Device>(0, _container->getNumCells()),
            KOKKOS_LAMBDA(uint_type cell_idx, uint_type& local_max) {
                const uint_type count = bin_offsets(cell_idx + 1) - bin_offsets(cell_idx);
                if (count > local_max) local_max = count;
            },
            Kokkos::Max<uint_type>(max_particles_per_cell)
        );

        // Precompute maximum number of particles in neighborhood to get scratch memory size
        uint_type max_neighborhood = 0;
        Kokkos::parallel_reduce(
            "lc_cell_max_neighborhood",
            Kokkos::MDRangePolicy<Device, Kokkos::Rank<3>>({1, 1, 1}, {num_cells_per_dim[0] - 1, num_cells_per_dim[1] - 1, num_cells_per_dim[2] - 1}),
            KOKKOS_LAMBDA(uint_type cell_x, uint_type cell_y, uint_type cell_z, uint_type& local_max) {
                uint_type count = 0;
                for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
                for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
                for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
                    const uint_type cell_idx_j = cell_indexing.encode({
                        static_cast<uint_type>(cell_x + delta_x),
                        static_cast<uint_type>(cell_y + delta_y),
                        static_cast<uint_type>(cell_z + delta_z)
                    });
                    count += bin_offsets(cell_idx_j + 1) - bin_offsets(cell_idx_j);
                }
                if (count > local_max) local_max = count;
            },
            Kokkos::Max<uint_type>(max_neighborhood)
        );

        using team_policy_t = Kokkos::TeamPolicy<Device>;
        using member_t = team_policy_t::member_type;
        using ScratchFloat = Kokkos::View<float_type*, Device::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using ScratchUint = Kokkos::View<uint_type*, Device::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // Byte count for the staged particles: one id array plus nine r/v/f arrays
        const size_t scratch_bytes = ScratchUint::shmem_size(max_neighborhood) + 9 * ScratchFloat::shmem_size(max_neighborhood);
        const int scratch_level = selectScratchLevel(scratch_bytes, "lc_cell_force_scratch", _warned_scratch_fallback);

        team_policy_t policy(static_cast<int>(league_size), static_cast<int>(std::clamp<uint_type>(max_particles_per_cell, 1, MAX_TEAM_SIZE)));
        policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes));

        Kokkos::parallel_for(
            "lc_cell_force_scratch", policy,
            KOKKOS_LAMBDA(const member_t& team) {
                // Get cell coords from team rank
                const uint_type league_rank = static_cast<uint_type>(team.league_rank());
                const uint_type cell_x = 1 + (league_rank % cells_x);
                const uint_type cell_y = 1 + ((league_rank / cells_x) % cells_y);
                const uint_type cell_z = 1 + (league_rank / (cells_x * cells_y));

                // Allocate scratch memory
                ScratchUint s_id(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_rx(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_ry(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_rz(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_vx(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_vy(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_vz(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_fx(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_fy(team.team_scratch(scratch_level), max_neighborhood);
                ScratchFloat s_fz(team.team_scratch(scratch_level), max_neighborhood);

                // Stage the home cell first
                const uint_type cell_idx_i = cell_indexing.encode({cell_x, cell_y, cell_z});
                const uint_type home_begin = bin_offsets(cell_idx_i);
                const uint_type home_count = bin_offsets(cell_idx_i + 1) - home_begin;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, home_count), [&](uint_type k) {
                    const uint_type src = home_begin + k;
                    s_id(k) = ids(src);
                    s_rx(k) = rx(src);
                    s_ry(k) = ry(src);
                    s_rz(k) = rz(src);
                    s_vx(k) = vx(src);
                    s_vy(k) = vy(src);
                    s_vz(k) = vz(src);
                    s_fx(k) = fx(src);
                    s_fy(k) = fy(src);
                    s_fz(k) = fz(src);
                });
                uint_type offset = home_count;

                // Stage neighbor cells
                for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
                for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
                for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
                    if (delta_x == 0 && delta_y == 0 && delta_z == 0) continue;  // home cell already staged
                    const uint_type cell_idx_j = cell_indexing.encode({
                        static_cast<uint_type>(cell_x + delta_x),
                        static_cast<uint_type>(cell_y + delta_y),
                        static_cast<uint_type>(cell_z + delta_z)
                    });
                    const uint_type j_begin = bin_offsets(cell_idx_j);
                    const uint_type j_count = bin_offsets(cell_idx_j + 1) - j_begin;
                    const uint_type dst = offset;
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, j_count), [&](uint_type k) {
                        const uint_type src = j_begin + k;
                        s_id(dst + k) = ids(src);
                        s_rx(dst + k) = rx(src);
                        s_ry(dst + k) = ry(src);
                        s_rz(dst + k) = rz(src);
                        s_vx(dst + k) = vx(src);
                        s_vy(dst + k) = vy(src);
                        s_vz(dst + k) = vz(src);
                        s_fx(dst + k) = fx(src);
                        s_fy(dst + k) = fy(src);
                        s_fz(dst + k) = fz(src);
                    });
                    offset += j_count;
                }
                const uint_type total_count = offset;

                team.team_barrier();  // staging complete before the force loop reads scratch

                // One thread per home particle. Sweep all staged particles and skip self (i == j).
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, home_count), [&](uint_type i) {
                    Particle particle_i;
                    particle_i.id = s_id(i);
                    particle_i.r = {s_rx(i), s_ry(i), s_rz(i)};
                    particle_i.v = {s_vx(i), s_vy(i), s_vz(i)};
                    particle_i.f = {s_fx(i), s_fy(i), s_fz(i)};

                    Vec3 force = {float_type(0), float_type(0), float_type(0)};
                    for (uint_type j = 0; j < total_count; ++j) {
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

                    const uint_type global_i = home_begin + i;
                    fx(global_i) += force[0];
                    fy(global_i) += force[1];
                    fz(global_i) += force[2];
                });
            });
        Kokkos::fence();
        _container->modifyDevice();
    }
};
