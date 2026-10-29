#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"
#include "ForceFunctorInterface.hpp"

/**
 * High-arithmetic pairwise force functor. Implements the ForceFunctorInterface CRTP
 * interface. Runs a fixed number of iterations to model a high arithmetic intensity.
 * The concrete result is meaningless and is chosen to not be optimized out by the compiler:
 *   - the accumulators are seeded from the pair's displacement (runtime input),
 *   - each iteration carries a dependency on the previous one and on the loop
 *     index, so iterations cannot be reordered or collapsed
 */
struct HighComputeFunctor : ForceFunctorInterface<HighComputeFunctor> {
    uint_type iterations;

    explicit HighComputeFunctor(uint_type iterations_ = 100) : iterations(iterations_) {}

    LC_HOST_DEVICE_INLINE
    Vec3 evaluate(const Particle& i, const Particle& j) const {
        const float_type dx = j.r[0] - i.r[0];
        const float_type dy = j.r[1] - i.r[1];
        const float_type dz = j.r[2] - i.r[2];

        // Loop-carried accumulators seeded from the runtime displacement.
        float_type ax = dx;
        float_type ay = dy;
        float_type az = dz;
        for (uint_type k = 0; k < iterations; ++k) {
            const float_type t = float_type(k) + float_type(1);
            ax = bounded((ax * ay + t) * az + dx);
            ay = bounded((ay * az + t) * ax + dy);
            az = bounded((az * ax + t) * ay + dz);
        }
        return {ax, ay, az};
    }

   private:
    // Maps any finite value into (-1, 1) so the chain cannot reach inf.
    LC_HOST_DEVICE_INLINE
    static float_type bounded(float_type x) {
        const float_type magnitude = x < float_type(0) ? -x : x;
        return x / (float_type(1) + magnitude);
    }
};
