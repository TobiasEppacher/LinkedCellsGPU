#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"
#include "ForceFunctorInterface.hpp"

/**
 * Lennard-Jones pairwise force functor. Implements the ForceFunctorInterface CRTP interface
 */
struct LennardJonesFunctor : ForceFunctorInterface<LennardJonesFunctor> {
    float_type epsilon;
    float_type sigma;

    float_type A = 0.0;
    float_type B = 0.0;

    LennardJonesFunctor(float_type epsilon_, float_type sigma_)
        : epsilon(epsilon_),
          sigma(sigma_) {
            float_type s6 = sigma * sigma * sigma * sigma * sigma * sigma;
            A = float_type(48) * epsilon * s6 * s6;
            B = float_type(24) * epsilon * s6;
          }

    /**
     * Returns the Lennard-Jones force that particle j exerts on particle i.
     * Does not modify either particle: the caller accumulates the returned
     * vector into i's force. The cutoff check is done by the traversal.
     */
    LC_HOST_DEVICE_INLINE
    Vec3 evaluate(const Particle& i, const Particle& j) const {
        float_type dx = i.r[0] - j.r[0];
        float_type dy = i.r[1] - j.r[1];
        float_type dz = i.r[2] - j.r[2];
        float_type r2 = dx * dx + dy * dy + dz * dz;

        float_type r2_inv = float_type(1) / r2;
        float_type r6_inv = r2_inv * r2_inv * r2_inv;
        float_type r12_inv = r6_inv * r6_inv;
        float_type f_scalar = (A * r12_inv - B * r6_inv) * r2_inv;

        return {f_scalar * dx, f_scalar * dy, f_scalar * dz};
    }
};
