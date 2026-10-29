#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"
#include "ForceFunctorInterface.hpp"

/**
 * Low-arithmetic pairwise force functor that reads every functor input of both
 * particles (id, r, v -> not force).
 * Cheap, but a traversal has to load more inputs then only the position,
 * so the measurement covers the full per-pair load.
 *
 * The concrete result is physically meaningless
 */
struct FullParticleFunctor : ForceFunctorInterface<FullParticleFunctor> {
    LC_HOST_DEVICE_INLINE
    Vec3 evaluate(const Particle& i, const Particle& j) const {
        const float_type weight = float_type(1) + float_type((i.id ^ j.id) & ID_MASK) * ID_SCALE;
        return {weight * (j.r[0] - i.r[0]) + (j.v[0] - i.v[0]),
                weight * (j.r[1] - i.r[1]) + (j.v[1] - i.v[1]),
                weight * (j.r[2] - i.r[2]) + (j.v[2] - i.v[2])};
    }

   private:
    // Holds the id-derived weight in [1, 2) whatever the ids are.
    static constexpr uint_type ID_MASK = 0xF;
    static constexpr float_type ID_SCALE = float_type(1) / float_type(16);
};
