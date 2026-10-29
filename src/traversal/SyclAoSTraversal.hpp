#pragma once

#include <sycl/sycl.hpp>

#include "../container/SyclAoSContainer.hpp"
#include "../particle/Particle.hpp"
#include "../util/MathUtils.hpp"

/**
 * SYCL particle-based traversal on an array-of-structs (AoS) container.
 * The `applyForEach` and `applyPairwise` methods are conceptually indentical
 * to the `SYCLSoATraversal`, but use the AoS memory layout in the
 * corresponding parallel constructs instead.
 */
class SyclAoSTraversal {
    SyclAoSContainer* _container;

   public:
    SyclAoSTraversal(SyclAoSContainer* container) : _container(container) {}

    // One work item per particle. The functor mutates the struct in place.
    template <typename Func>
    void applyForEach(Func func) {
        _container->syncDevice();
        const uint_type n = _container->size();
        if (n > 0) {
            const SyclAoS p = _container->_aos;
            _container->_queue.parallel_for(
                sycl::nd_range<1>{sycl::range<1>{syclGlobalSize(n)},
                                  sycl::range<1>{SYCL_WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> item) {
                    const uint_type i = static_cast<uint_type>(item.get_global_id(0));
                    if (i >= n) return;
                    func(p.particles[i]);
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
            const SyclAoS p = _container->_aos;
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

                    const Particle particle_i = p.particles[i];

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
                            const Particle particle_j = p.particles[j];
                            if (dist_sq(particle_i.r, particle_j.r) > interaction_radius_sq) continue;
                            const Vec3 f = func(particle_i, particle_j);
                            force[0] += f[0];
                            force[1] += f[1];
                            force[2] += f[2];
                        }
                    }

                    p.particles[i].f[0] += force[0];
                    p.particles[i].f[1] += force[1];
                    p.particles[i].f[2] += force[2];
                });
        }
        _container->_queue.wait();
        _container->modifyDevice();
    }
};
