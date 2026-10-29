#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"
#include "ForceFunctorInterface.hpp"

/**
 * Low-arithmetic pairwise force functor. Implements the ForceFunctorInterface CRTP
 * interface. Does as little work possible while still depending on both input
 * particles so it isn't optimized away.
 */
struct LowComputeFunctor : ForceFunctorInterface<LowComputeFunctor> {
    LC_HOST_DEVICE_INLINE
    Vec3 evaluate(const Particle& i, const Particle& j) const {
        return {j.r[0] - i.r[0], j.r[1] - i.r[1], j.r[2] - i.r[2]};
    }
};
