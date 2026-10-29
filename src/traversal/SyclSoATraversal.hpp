#pragma once

#include <sycl/sycl.hpp>

#include "../container/SyclSoAContainer.hpp"
#include "../particle/Particle.hpp"
#include "../util/MathUtils.hpp"

/**
 * SYCL particle-based traversal on a struct-of-arrays (SoA) container.
 */
class SyclSoATraversal {
    SyclSoAContainer* _container;

   public:
    SyclSoATraversal(SyclSoAContainer* container) : _container(container) {}

    // One work item per particle. Materialised from the SoA buffers and written back after the call.
    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        const uint_type n = _container->size();
        if (n > 0) {
            const SyclSoA p = _container->_soa;
            _container->_queue.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)},
                                  sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> item) {
                    const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                    if (i >= n) return;
                    Particle particle_i;
                    particle_i.id = p.id[i];
                    particle_i.r = {p.rx[i], p.ry[i], p.rz[i]};
                    particle_i.v = {p.vx[i], p.vy[i], p.vz[i]};
                    particle_i.f = {p.fx[i], p.fy[i], p.fz[i]};
                    func(particle_i);
                    p.id[i] = particle_i.id;
                    p.rx[i] = particle_i.r[0];
                    p.ry[i] = particle_i.r[1];
                    p.rz[i] = particle_i.r[2];
                    p.vx[i] = particle_i.v[0];
                    p.vy[i] = particle_i.v[1];
                    p.vz[i] = particle_i.v[2];
                    p.fx[i] = particle_i.f[0];
                    p.fy[i] = particle_i.f[1];
                    p.fz[i] = particle_i.f[2];
                });
        }
        _container->_queue.wait();
        _container->modifyDevice();
    }

    /**
     * Linked-cells force computation.
     * Requires an up-to-date binning (i.e. containers rebuild() called first)
     */
    template <typename Func>
    void applyPairwise(Func func) {
        const uint_type n = _container->size();
        if (n > 0) {
            const SyclSoA p = _container->_soa;
            const uint_type* bin_offsets = _container->getBinOffsets();
            const float_type interaction_radius_sq =
                _container->getInteractionRadius() * _container->getInteractionRadius();
            const CellIndex cell_indexing = _container->_cell_indexing;

            _container->_queue.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)},
                                  sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> item) {
                    const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                    if (i >= n) return;

                    Particle particle_i;
                    particle_i.id = p.id[i];
                    particle_i.r = {p.rx[i], p.ry[i], p.rz[i]};
                    particle_i.v = {p.vx[i], p.vy[i], p.vz[i]};
                    particle_i.f = {p.fx[i], p.fy[i], p.fz[i]};

                    const Vec3u cell_coords_i = cell_indexing.decode(p.cell_idx[i]);

                    Vec3 force = {float_type(0), float_type(0), float_type(0)};
                    for (int_type delta_z = -1; delta_z <= 1; ++delta_z)
                    for (int_type delta_y = -1; delta_y <= 1; ++delta_y)
                    for (int_type delta_x = -1; delta_x <= 1; ++delta_x) {
                        const uint_type cell_idx_j = cell_indexing.encode({
                            static_cast<uint_type>(cell_coords_i[0] + delta_x),
                            static_cast<uint_type>(cell_coords_i[1] + delta_y),
                            static_cast<uint_type>(cell_coords_i[2] + delta_z)});
                        const uint_type j_begin = bin_offsets[cell_idx_j];
                        const uint_type j_end = bin_offsets[cell_idx_j + 1];
                        for (uint_type j = j_begin; j < j_end; ++j) {
                            if (j == i) continue;
                            Particle particle_j;
                            particle_j.id = p.id[j];
                            particle_j.r = {p.rx[j], p.ry[j], p.rz[j]};
                            particle_j.v = {p.vx[j], p.vy[j], p.vz[j]};
                            particle_j.f = {p.fx[j], p.fy[j], p.fz[j]};
                            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                            const Vec3 f = func(particle_i, particle_j);
                            force[0] += f[0];
                            force[1] += f[1];
                            force[2] += f[2];
                        }
                    }

                    p.fx[i] += force[0];
                    p.fy[i] += force[1];
                    p.fz[i] += force[2];
                });
        }
        _container->_queue.wait();
        _container->modifyDevice();
    }
};
