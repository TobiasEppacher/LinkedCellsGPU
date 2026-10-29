#pragma once

#include <stdexcept>

#include "../io/InitializerParams.hpp"
#include "../particle/Particle.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Places particles on a regular grid filling [box_min, box_max]. grid_counts holds the
 * number of particles along each axis; the first and last of each axis sit exactly on
 * the box corners, so a cuboid spanning the whole domain puts particles on its
 * boundary. Ids are assigned sequentially (x fastest); velocities and forces keep the
 * Particle defaults (zero).
 */
class CuboidInitializer {
    CuboidInitializerParams params;
    Vec3 spacing;

   public:
    explicit CuboidInitializer(const CuboidInitializerParams& init_params) : params(init_params) {
        const Vec3u& counts = params.grid_counts;
        if (counts[0] < 2 || counts[1] < 2 || counts[2] < 2) {
            throw std::invalid_argument("CuboidInitializer: grid_counts must be at least 2 along each axis");
        }
        spacing = {
            axisSpacing(params.box_min[0], params.box_max[0], counts[0]),
            axisSpacing(params.box_min[1], params.box_max[1], counts[1]),
            axisSpacing(params.box_min[2], params.box_max[2], counts[2])
        };
    }

    // Returns the first id not used by this initializer.
    template <typename Container>
    uint_type initialize(Container& container, uint_type first_id = 0) const {
        uint_type id = first_id;
        for (uint_type iz = 0; iz < params.grid_counts[2]; ++iz) {
            for (uint_type iy = 0; iy < params.grid_counts[1]; ++iy) {
                for (uint_type ix = 0; ix < params.grid_counts[0]; ++ix) {
                    Particle p;
                    p.id = id++;
                    p.r = {
                        params.box_min[0] + ix * spacing[0],
                        params.box_min[1] + iy * spacing[1],
                        params.box_min[2] + iz * spacing[2]
                    };
                    container.addParticle(p);
                }
            }
        }
        return id;
    }

   private:
    static float_type axisSpacing(float_type min, float_type max, uint_type count) {
        return (max - min) / static_cast<float_type>(count - 1);
    }
};
