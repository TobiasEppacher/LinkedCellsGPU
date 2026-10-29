#pragma once

#include "../particle/Particle.hpp"
#include "../util/TypeDefinitions.hpp"

struct ParticleContainer {
    virtual ~ParticleContainer() = default;
    virtual void addParticle(const Particle& p) = 0;
    virtual uint_type size() const = 0;
    virtual void rebuild() = 0;
};