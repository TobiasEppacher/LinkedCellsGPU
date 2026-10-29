#pragma once

#include <cstddef>
#include <sstream>
#include <string>

#include "../util/TypeDefinitions.hpp"

/**
 * @brief Parameters for the Random initializer.
 */
struct RandomInitializerParams {
    /**
     * @brief The number of particles to generate randomly within the specified box.
     */
    uint_type num_particles;

    /**
     * @brief The minimum corner of the box in which to randomly generate particles.
     */
    Vec3 box_min;

    /**
     * @brief The maximum corner of the box in which to randomly generate particles.
     */
    Vec3 box_max;

    /**
     * @brief The number of particles this initializer seeds.
     */
    size_t particleCount() const { return static_cast<size_t>(num_particles); }

    std::string toString() const {
        std::ostringstream out;
        out << "random (" << particleCount() << " particles in [" << box_min[0] << ", " << box_min[1]
            << ", " << box_min[2] << "] .. [" << box_max[0] << ", " << box_max[1] << ", " << box_max[2]
            << "])";
        return out.str();
    }
};

/**
 * @brief Parameters for the Cuboid initializer.
 */
struct CuboidInitializerParams {
    /**
     * @brief The minimum corner of the cuboid in which to generate particles.
     */
    Vec3 box_min;

    /**
     * @brief The maximum corner of the cuboid in which to generate particles.
     */
    Vec3 box_max;

    /**
     * @brief The number of particles along each axis.
     */
    Vec3u grid_counts;

    /**
     * @brief The number of particles this initializer seeds.
     */
    size_t particleCount() const {
        return static_cast<size_t>(grid_counts[0]) * grid_counts[1] * grid_counts[2];
    }

    std::string toString() const {
        std::ostringstream out;
        out << "cuboid (" << grid_counts[0] << " x " << grid_counts[1] << " x " << grid_counts[2]
            << " = " << particleCount() << " particles in [" << box_min[0] << ", " << box_min[1] << ", "
            << box_min[2] << "] .. [" << box_max[0] << ", " << box_max[1] << ", " << box_max[2] << "])";
        return out.str();
    }
};
