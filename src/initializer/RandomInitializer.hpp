#pragma once

#include <random>

#include "../io/InitializerParams.hpp"
#include "../particle/Particle.hpp"
#include "../util/TypeDefinitions.hpp"

/**
 * Fills a container with num_particles particles at uniformly random positions
 * inside [box_min, box_max). Ids are assigned sequentially; velocities and forces
 * keep the Particle defaults (zero). This is the scheme the simulation seeded with
 * before initializers were factored out.
 */
class RandomInitializer {
    RandomInitializerParams params;

   public:
    explicit RandomInitializer(const RandomInitializerParams& init_params) : params(init_params) {}

    // Returns the first id not used by this initializer.
    template <typename Container>
    uint_type initialize(Container& container, uint_type first_id = 0) const {
        uint_type id = first_id;
        for (uint_type i = 0; i < params.num_particles; ++i) {
            Particle p;
            p.id = id++;
            p.r = {randomInAxis(0), randomInAxis(1), randomInAxis(2)};
            container.addParticle(p);
        }
        return id;
    }

   private:
    // Shared across initializers so multiple random entries don't draw identical positions.
    static std::mt19937& engine() {
        static std::mt19937 gen(42);
        return gen;
    }

    // Uniform sample in [box_min[d], box_max[d])
    float_type randomInAxis(size_t d) const {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        return static_cast<float_type>(unit(engine())) * (params.box_max[d] - params.box_min[d]) + params.box_min[d];
    }
};
