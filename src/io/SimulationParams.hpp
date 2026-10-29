#pragma once

#include "../util/TypeDefinitions.hpp"

#include "AlgorithmType.hpp"
#include "FunctorType.hpp"
#include "InitializerParams.hpp"

#include <cstddef>
#include <filesystem>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

/**
 * @brief The parameters for a simulation run.
 */
struct SimulationParams {
    /**
     * @brief The path to the input file containing the simulation parameters.
     */
    std::filesystem::path input_file_path;

    /**
     * @brief The path to the output directory where simulation results will be saved.
     */
    std::filesystem::path output_dir;

    /**
     * @brief The time step size for the simulation.
     */
    float_type delta_t;

    /**
     * @brief The total simulation time.
     */
    float_type end_t;

    /**
     * @brief The size of the simulation domain in each dimension (x, y, z).
     */
    Vec3 domain_size;

    /**
     * @brief The interaction radius for the particles in the simulation.
     */
    float_type interaction_radius;

    /**
     * @brief The depth of the Lennard-Jones potential well.
     */
    float_type epsilon;

    /**
     * @brief The zero-crossing distance of the Lennard-Jones potential.
     */
    float_type sigma;

    /**
     * @brief The selected algorithm for the simulation
     */
    AlgorithmType algorithm_type;

    /**
     * @brief The selected pairwise force functor for the simulation.
     */
    FunctorType functor_type;

    /**
     * @brief Whether an iteration only recomputes the forces instead of running a full timestep.
     */
    bool only_forces;

    /**
     * @brief Whether a VTK snapshot is written per timestep (requires an output directory).
     */
    bool output_vtk_files;

    /**
     * @brief A vector of initializers for the simulation.
     */
    std::vector<std::variant<RandomInitializerParams, CuboidInitializerParams>> initializers;

    SimulationParams() = default;

    /**
     * @brief The number of particles seeded by all initializers together.
     */
    size_t totalParticleCount() const {
        size_t count = 0;
        for (const auto& initializer : initializers) {
            count += std::visit([](const auto& params) { return params.particleCount(); }, initializer);
        }
        return count;
    }

    /**
     * @brief Human-readable dump of the configuration for the console/log.
     */
    std::string toString() const {
        std::ostringstream out;
        out << "\n--- Configuration ---\n\n"
            << "Config file: " << input_file_path.string() << "\n"
            << "Output directory: " << (output_dir.empty() ? "(none)" : output_dir.string()) << "\n"
            << "Timestep size: " << delta_t << "\n"
            << "End time: " << end_t << "\n"
            << "Domain size: [" << domain_size[0] << ", " << domain_size[1] << ", " << domain_size[2]
            << "]\n"
            << "Interaction radius: " << interaction_radius << "\n"
            << "Epsilon: " << epsilon << "\n"
            << "Sigma: " << sigma << "\n"
            << "Algorithm: " << getKeyFromAlgorithmType(algorithm_type) << "\n"
            << "Functor: " << getKeyFromFunctorType(functor_type) << "\n"
            << "Force calculation only: " << (only_forces ? "yes" : "no") << "\n"
            << "VTK output: " << (output_vtk_files ? "on" : "off") << "\n"
            << "Number of particles: " << totalParticleCount() << "\n"
            << "Initializers:\n";
        for (const auto& initializer : initializers) {
            out << "  - " << std::visit([](const auto& params) { return params.toString(); }, initializer)
                << "\n";
        }
        out << "\n";
        return out.str();
    }
};
