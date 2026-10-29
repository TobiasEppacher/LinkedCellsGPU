#pragma once

#include "../container/KokkosSoAContainer.hpp"
#include "../util/KokkosDefinitions.hpp"
#include "../util/MathUtils.hpp"

/**
 * Kokkos particle-based traversal on a struct-of-arrays (SoA) container.
 */
class KokkosSoATraversal {
    KokkosSoAContainer* _container;

    public:
    KokkosSoATraversal(KokkosSoAContainer* container) : _container(container) {}

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

        // Materialise the particle from the parallel arrays, apply the functor, write it back.
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
        const uint_type num_particles = _container->size();
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
        auto cell_idx = _container->_soa.cell_idx;
        const CellIndex cell_indexing = _container->_cell_indexing;

        Kokkos::parallel_for(
            "lc_force", Kokkos::RangePolicy<Device>(0, num_particles),
            KOKKOS_LAMBDA(uint_type i) {
                Particle particle_i;
                particle_i.id = ids(i);
                particle_i.r = {rx(i), ry(i), rz(i)};
                particle_i.v = {vx(i), vy(i), vz(i)};
                particle_i.f = {fx(i), fy(i), fz(i)};

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
        Kokkos::fence();
        _container->modifyDevice();
    }
};
