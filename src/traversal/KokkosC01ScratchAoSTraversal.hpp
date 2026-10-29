#pragma once

#include <algorithm>

#include "../container/KokkosAoSContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/KokkosScratch.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos cell-grouped particle-based traversal on an array-of-structs (AoS) container.
 * The `applyForEach` and `applyPairwise` methods are conceptually indentical
 * to the `KokkosC01ScratchSoATraversal`, but use the AoS memory layout in the corresponding parallel constructs instead.
 */
class KokkosC01ScratchAoSTraversal {
    KokkosAoSContainer* _container;
    bool _warned_scratch_fallback = false;

    public:
    KokkosC01ScratchAoSTraversal(KokkosAoSContainer* container)
        : _container(container) {}

    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        auto particles = _container->_aos.particles;

        // The functor mutates the particle in place, so the struct is updated where it lies.
        Kokkos::parallel_for(
            "forEachParticle", Kokkos::RangePolicy<Device>(0, _container->size()),
            KOKKOS_LAMBDA(uint_type i) { func(particles(i)); });
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
        auto particles = _container->_aos.particles;
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
        using ScratchParticle = Kokkos::View<Particle*, Device::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // Byte count for the staged particles: one array of structs
        const size_t scratch_bytes = ScratchParticle::shmem_size(max_neighborhood);
        const int scratch_level = selectScratchLevel(scratch_bytes, "lc_cell_force_scratch_aos", _warned_scratch_fallback);

        team_policy_t policy(static_cast<int>(league_size), static_cast<int>(std::clamp<uint_type>(max_particles_per_cell, 1, MAX_TEAM_SIZE)));
        policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes));

        Kokkos::parallel_for(
            "lc_cell_force_scratch_aos", policy,
            KOKKOS_LAMBDA(const member_t& team) {
                // Get cell coords from team rank
                const uint_type league_rank = static_cast<uint_type>(team.league_rank());
                const uint_type cell_x = 1 + (league_rank % cells_x);
                const uint_type cell_y = 1 + ((league_rank / cells_x) % cells_y);
                const uint_type cell_z = 1 + (league_rank / (cells_x * cells_y));

                // Allocate scratch memory
                ScratchParticle s_particles(team.team_scratch(scratch_level), max_neighborhood);

                // Stage the home cell first
                const uint_type cell_idx_i = cell_indexing.encode({cell_x, cell_y, cell_z});
                const uint_type home_begin = bin_offsets(cell_idx_i);
                const uint_type home_count = bin_offsets(cell_idx_i + 1) - home_begin;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, home_count), [&](uint_type k) {
                    s_particles(k) = particles(home_begin + k);
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
                        s_particles(dst + k) = particles(j_begin + k);
                    });
                    offset += j_count;
                }
                const uint_type total_count = offset;

                team.team_barrier();  // staging complete before the force loop reads scratch

                // One thread per home particle. Sweep all staged particles and skip self (i == j).
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, home_count), [&](uint_type i) {
                    const Particle particle_i = s_particles(i);

                    Vec3 force = {float_type(0), float_type(0), float_type(0)};
                    for (uint_type j = 0; j < total_count; ++j) {
                        if (j == i) continue;
                        const Particle particle_j = s_particles(j);
                        if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                        const Vec3 f = func(particle_i, particle_j);
                        force[0] += f[0];
                        force[1] += f[1];
                        force[2] += f[2];
                    }

                    const uint_type global_i = home_begin + i;
                    particles(global_i).f[0] += force[0];
                    particles(global_i).f[1] += force[1];
                    particles(global_i).f[2] += force[2];
                });
            });
        Kokkos::fence();
        _container->modifyDevice();
    }
};
