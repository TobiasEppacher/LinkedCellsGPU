#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>

#include "../container/CpuAoSContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/MathUtils.hpp"

class CpuTraversal {
    CpuAoSContainer* _container;

  public:
    CpuTraversal(CpuAoSContainer* container) : _container(container) {}

    /**
     * C01 traversal on host: each cell interacts with itself and all 26 neighbors.
     * Parallelised over cells; func returns the force on p1 (the base-cell
     * particle) and the traversal accumulates it only into p1, so it is race-free.
     * Requires an up-to-date cell binning: call the container's rebuild() first.
     */
    template <typename Func>
    void applyPairwiseSerial(Func func) {
        const Vec3u d = _container->_num_cells_with_halo;
        const float_type interaction_radius_sq = _container->getInteractionRadius() * _container->getInteractionRadius();

        Kokkos::parallel_for("C01Serial",
                            Kokkos::MDRangePolicy<Serial, Kokkos::Rank<3>>({1, 1, 1}, {d[0] - 1, d[1] - 1, d[2] - 1}),
                            KOKKOS_CLASS_LAMBDA(uint_type ci, uint_type cj, uint_type ck) {
                                AoSCell& base_cell = _container->getCell(ci, cj, ck);
                                const uint_type base_size = base_cell.size();

                                constexpr int hs[27][3] = {{0, 0, 0}, {-1, -1, -1}, {0, -1, -1}, {1, -1, -1}, {-1, 0, -1}, {0, 0, -1}, {1, 0, -1}, {-1, 1, -1},
                                                        {0, 1, -1},   {1, 1, -1},  {-1, -1, 0}, {0, -1, 0},  {1, -1, 0}, {-1, 0, 0}, {1, 0, 0},
                                                        {-1, 1, 0},   {0, 1, 0},   {1, 1, 0},   {-1, -1, 1}, {0, -1, 1}, {1, -1, 1}, {-1, 0, 1},
                                                        {0, 0, 1},    {1, 0, 1},   {-1, 1, 1},  {0, 1, 1},   {1, 1, 1}};

                                // Full-shell neighbor interactions
                                for (int n = 0; n < 27; ++n) {
                                    AoSCell& neighbor_cell = _container->getCell(ci + hs[n][0], cj + hs[n][1], ck + hs[n][2]);
                                    const uint_type neighbor_size = neighbor_cell.size();
                                    for (uint_type i = 0; i < base_size; ++i) {
                                        Particle& particle_i = base_cell.data(i);
                                        for (uint_type j = 0; j < neighbor_size; ++j) {
                                            if (n == 0 && i == j) continue;  // skip self only within the home cell; i,j are per-cell indices
                                            Particle& particle_j = neighbor_cell.data(j);
                                            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                                            const Vec3 force = func(particle_i, particle_j);
                                            particle_i.f[0] += force[0];
                                            particle_i.f[1] += force[1];
                                            particle_i.f[2] += force[2];
                                        }
                                    }
                                }
                            });
        Kokkos::fence();
    }

    template<typename Func>
    void applyForEach(Func func) {
        const uint_type num_cells = _container->_cells.size();

        Kokkos::parallel_for("forEachParticle", Kokkos::RangePolicy<Host>(0, num_cells),
            KOKKOS_CLASS_LAMBDA(uint_type cell_idx) {
                AoSCell& cell = _container->_cells[cell_idx];
                for (uint_type i = 0; i < cell.size(); ++i) {
                    func(cell.data(i));
                }
            }
        );
        Kokkos::fence();
    }

    /**
     * C01 traversal on host: each cell interacts with itself and all 26 neighbors.
     * Parallelised over cells; func returns the force on p1 (the base-cell
     * particle) and the traversal accumulates it only into p1, so it is race-free.
     * Requires an up-to-date cell binning: call the container's rebuild() first.
     */
    template <typename Func>
    void applyPairwise(Func func) {
        const Vec3u d = _container->_num_cells_with_halo;
        const float_type interaction_radius_sq = _container->getInteractionRadius() * _container->getInteractionRadius();

        Kokkos::parallel_for("C01Host",
                            Kokkos::MDRangePolicy<Host, Kokkos::Rank<3>>({1, 1, 1}, {d[0] - 1, d[1] - 1, d[2] - 1}),
                            KOKKOS_CLASS_LAMBDA(uint_type ci, uint_type cj, uint_type ck) {
                                AoSCell& base_cell = _container->getCell(ci, cj, ck);
                                const uint_type base_size = base_cell.size();

                                constexpr int hs[27][3] = {{0, 0, 0}, {-1, -1, -1}, {0, -1, -1}, {1, -1, -1}, {-1, 0, -1}, {0, 0, -1}, {1, 0, -1}, {-1, 1, -1},
                                                        {0, 1, -1},   {1, 1, -1},  {-1, -1, 0}, {0, -1, 0},  {1, -1, 0}, {-1, 0, 0}, {1, 0, 0},
                                                        {-1, 1, 0},   {0, 1, 0},   {1, 1, 0},   {-1, -1, 1}, {0, -1, 1}, {1, -1, 1}, {-1, 0, 1},
                                                        {0, 0, 1},    {1, 0, 1},   {-1, 1, 1},  {0, 1, 1},   {1, 1, 1}};

                                // Full-shell neighbor interactions
                                for (int n = 0; n < 27; ++n) {
                                    AoSCell& neighbor_cell = _container->getCell(ci + hs[n][0], cj + hs[n][1], ck + hs[n][2]);
                                    const uint_type neighbor_size = neighbor_cell.size();
                                    for (uint_type i = 0; i < base_size; ++i) {
                                        Particle& particle_i = base_cell.data(i);
                                        for (uint_type j = 0; j < neighbor_size; ++j) {
                                            if (n == 0 && i == j) continue;  // skip self only within the home cell; i,j are per-cell indices
                                            Particle& particle_j = neighbor_cell.data(j);
                                            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                                            const Vec3 force = func(particle_i, particle_j);
                                            particle_i.f[0] += force[0];
                                            particle_i.f[1] += force[1];
                                            particle_i.f[2] += force[2];
                                        }
                                    }
                                }
                            });
        Kokkos::fence();
    }
};
