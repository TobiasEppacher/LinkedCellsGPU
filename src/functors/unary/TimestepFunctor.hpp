#pragma once

#include "UnaryFunctorInterface.hpp"
#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"

/**
 * Performs one explicit-Euler integration step (mass = 1) followed by simple
 * reflecting boundaries against [box_min, box_max). Leaves the force untouched;
 * apply ResetForceFunctor before the next force calculation.
 */
struct TimestepFunctor : UnaryFunctorInterface<TimestepFunctor> {
    /**
     * Keeps a reflection off box_max, which maps the boundary onto itself, out of the
     * halo cell. Scaled by the box: an absolute epsilon vanishes in float rounding for
     * domains of ~64 and larger.
     */
    static constexpr float_type BOUNDARY_MARGIN = float_type(1e-6);

    float_type dt;
    Vec3 box_min;
    Vec3 box_max;

    TimestepFunctor(float_type timestep, const Vec3& min, const Vec3& max)
        : dt(timestep), box_min(min), box_max(max) {}

    LC_HOST_DEVICE_INLINE
    void evaluate(Particle& p) const {
        for (int d = 0; d < 3; ++d) {
            p.v[d] += p.f[d] * dt;
            p.r[d] += p.v[d] * dt;

            if (p.r[d] < box_min[d]) {
                p.r[d] = float_type(2.0) * box_min[d] - p.r[d];
                p.v[d] = -p.v[d];
            } else if (p.r[d] >= box_max[d]) {
                p.r[d] = float_type(2.0) * box_max[d] - p.r[d] -
                         BOUNDARY_MARGIN * (box_max[d] - box_min[d]);
                p.v[d] = -p.v[d];
            }
        }
    }
};

/**
 * Resets the force vector of a particle to zero. Used after each timestep to
 * prepare for the next force calculation.
 */
struct ResetForceFunctor : UnaryFunctorInterface<ResetForceFunctor> {
    LC_HOST_DEVICE_INLINE
    void evaluate(Particle& p) const {
        for (int d = 0; d < 3; ++d) p.f[d] = float_type(0.0);
    }
};
