#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"

/**
 * Static (CRTP) interface every pairwise force functor implements.
 * Uses CRTP to resolve the evaluate() at compile time.
 * Functors thus can be used in device code without virtual dispatch
 */
template <typename Derived>
struct ForceFunctorInterface {
    /**
     * Evaluation operator: returns the force vector exerted on particle i by
     * particle j. Forwards to the derived functor's evaluate()
     * Must not read i.f or j.f: forces are written concurrently without synchronization.
     *
     * @param i The particle on which the force is exerted
     * @param j The particle exerting the force
     * @return The force vector exerted on particle i by particle j
     */
    LC_HOST_DEVICE_INLINE
    Vec3 operator()(const Particle& i, const Particle& j) const {
        return static_cast<const Derived&>(*this).evaluate(i, j);
    }
};
