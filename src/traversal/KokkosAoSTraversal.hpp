#pragma once

#include "../container/KokkosAoSContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos particle-based traversal on an array-of-structs (AoS) container.
 * The `applyForEach` and `applyPairwise` methods are conceptually indentical
 * to the `KokkosSoATraversal`, but use the AoS memory layout in the
 * corresponding parallel constructs instead.
 */
class KokkosAoSTraversal {
    KokkosAoSContainer* _container;

    public:
    KokkosAoSTraversal(KokkosAoSContainer* container) : _container(container) {}

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
        const uint_type num_particles = _container->size();
        const auto bin_offsets = _container->getBinOffsets();
        const float_type interaction_radius_sq = _container->getInteractionRadius() * _container->getInteractionRadius();

        // Extract device views for kernel to capture.
        // Dereference of _container is invalid on device
        auto particles = _container->_aos.particles;
        auto cell_idx = _container->_aos.cell_idx;
        const CellIndex cell_indexing = _container->_cell_indexing;

        Kokkos::parallel_for(
            "lc_force_aos", Kokkos::RangePolicy<Device>(0, num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                const Particle particle_i = particles(i);

                const Vec3u cell_coords_i = cell_indexing.decode(cell_idx(i));

                Vec3 force = {float_type(0), float_type(0), float_type(0)};
                for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
                for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
                for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
                    const uint_type cell_idx_j = cell_indexing.encode({
                        static_cast<uint_type>(cell_coords_i[0] + delta_x),
                        static_cast<uint_type>(cell_coords_i[1] + delta_y),
                        static_cast<uint_type>(cell_coords_i[2] + delta_z)
                    });
                    const uint_type j_begin = bin_offsets(cell_idx_j);
                    const uint_type j_end = bin_offsets(cell_idx_j + 1);
                    for (uint_type j = j_begin; j < j_end; ++j) {
                        if (j == i) continue;
                        const Particle particle_j = particles(j);
                        if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                        const Vec3 f = func(particle_i, particle_j);
                        force[0] += f[0];
                        force[1] += f[1];
                        force[2] += f[2];
                    }
                }

                particles(i).f[0] += force[0];
                particles(i).f[1] += force[1];
                particles(i).f[2] += force[2];
            });
        Kokkos::fence();
        _container->modifyDevice();
    }
};
