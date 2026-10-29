#pragma once

#include <algorithm>

#include "../container/KokkosSoAContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos cell-grouped particle-based traversal on a struct-of-arrays (SoA) container.
 * The grouping of threads-per-particle into cells is done via a Kokkos::TeamPolicy
 * for each inner cell. Each thread computes the full force on one particle of the
 * associated home cell in a particle-based manner over the 27-cell neighbourhood,
 * known from the C01 traversal.
 */
class KokkosC01SoATraversal {
    KokkosSoAContainer* _container;

    public:
    KokkosC01SoATraversal(KokkosSoAContainer* container)
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

        using team_policy_t = Kokkos::TeamPolicy<Device>;
        using member_t = team_policy_t::member_type;

        team_policy_t policy(static_cast<int>(league_size), static_cast<int>(std::clamp<uint_type>(max_particles_per_cell, 1, MAX_TEAM_SIZE)));

        Kokkos::parallel_for(
            "lc_cell_force", policy,
            KOKKOS_LAMBDA(const member_t& team) {
                // Get cell coords from team rank
                const uint_type league_rank = static_cast<uint_type>(team.league_rank());
                const uint_type cell_x = 1 + (league_rank % cells_x);
                const uint_type cell_y = 1 + ((league_rank / cells_x) % cells_y);
                const uint_type cell_z = 1 + (league_rank / (cells_x * cells_y));

                const uint_type cell_idx_i = cell_indexing.encode({cell_x, cell_y, cell_z});
                const uint_type home_begin = bin_offsets(cell_idx_i);
                const uint_type home_count = bin_offsets(cell_idx_i + 1) - home_begin;

                // One thread per home particle, sweeping the neighbourhood from global memory.
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, home_count), [&](uint_type i) {
                    const uint_type global_i = home_begin + i;
                    Particle particle_i;
                    particle_i.id = ids(global_i);
                    particle_i.r = {rx(global_i), ry(global_i), rz(global_i)};
                    particle_i.v = {vx(global_i), vy(global_i), vz(global_i)};
                    particle_i.f = {fx(global_i), fy(global_i), fz(global_i)};

                    Vec3 force = {float_type(0), float_type(0), float_type(0)};
                    for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
                    for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
                    for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
                        const uint_type cell_idx_j = cell_indexing.encode({
                            static_cast<uint_type>(cell_x + delta_x),
                            static_cast<uint_type>(cell_y + delta_y),
                            static_cast<uint_type>(cell_z + delta_z)
                        });
                        const uint_type j_begin = bin_offsets(cell_idx_j);
                        const uint_type j_end = bin_offsets(cell_idx_j + 1);
                        for (uint_type j = j_begin; j < j_end; ++j) {
                            if (j == global_i) continue;
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

                    fx(global_i) += force[0];
                    fy(global_i) += force[1];
                    fz(global_i) += force[2];
                });
            });
        Kokkos::fence();
        _container->modifyDevice();
    }
};
