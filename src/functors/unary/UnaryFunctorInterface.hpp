#pragma once

#include "../../particle/Particle.hpp"
#include "../../util/DeviceDefinitions.hpp"
#include "../../util/TypeDefinitions.hpp"

/**
 * Static (CRTP) interface every unary particle functor implements.
 * Uses CRTP to resolve the evaluate() at compile time.
 * Functors thus can be used in device code without virtual dispatch
 */
template <typename Derived>
struct UnaryFunctorInterface {
    /**
     * Unary operator: Performs some action on a particle, modifying it in place.
     * Forwards to the derived functor's evaluate()
     *
     * @param p The particle on which this functor acts
     */
    LC_HOST_DEVICE_INLINE
    void operator()(Particle& p) const {
        static_cast<const Derived&>(*this).evaluate(p);
    }
};