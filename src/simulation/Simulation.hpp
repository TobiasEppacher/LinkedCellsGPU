#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>

#include "../functors/pairwise_force/FullParticleFunctor.hpp"
#include "../functors/pairwise_force/HighComputeFunctor.hpp"
#include "../functors/pairwise_force/LennardJonesFunctor.hpp"
#include "../functors/pairwise_force/LowComputeFunctor.hpp"
#include "../functors/unary/TimestepFunctor.hpp"
#include "../initializer/CuboidInitializer.hpp"
#include "../initializer/RandomInitializer.hpp"
#include "../io/SimulationParams.hpp"
#include "../util/Profiler.hpp"
#include "../util/VTKWriter.hpp"

// Backend specific includes
#if defined(LC_BACKEND_KOKKOS)
#include <Kokkos_Core.hpp>

#include "../container/CpuAoSContainer.hpp"
#include "../container/KokkosAoSContainer.hpp"
#include "../container/KokkosSoAContainer.hpp"
#include "../traversal/CpuTraversal.hpp"
#include "../traversal/KokkosAoSTraversal.hpp"
#include "../traversal/KokkosC01AoSTraversal.hpp"
#include "../traversal/KokkosC01ScratchAoSTraversal.hpp"
#include "../traversal/KokkosC01ScratchSoATraversal.hpp"
#include "../traversal/KokkosC01SoATraversal.hpp"
#include "../traversal/KokkosC08ScratchSoATraversal.hpp"
#include "../traversal/KokkosC08SoATraversal.hpp"
#include "../traversal/KokkosSoANewton3Traversal.hpp"
#include "../traversal/KokkosSoATraversal.hpp"
#elif defined(LC_BACKEND_CUDA)
#include "../container/CudaAoSContainer.hpp"
#include "../container/CudaSoAContainer.hpp"
#include "../traversal/CudaAoSTraversal.hpp"
#include "../traversal/CudaSoATraversal.hpp"
#elif defined(LC_BACKEND_SYCL)
#include "../container/SyclAoSContainer.hpp"
#include "../container/SyclSoAContainer.hpp"
#include "../traversal/SyclAoSTraversal.hpp"
#include "../traversal/SyclSoATraversal.hpp"
#else
#error "No backend selected: define LC_BACKEND_KOKKOS, LC_BACKEND_CUDA or LC_BACKEND_SYCL"
#endif

/**
 * Simulation class that orchestrates the particle simulation using its given SimulationParams.
 * Handles initialization and execution.
 * Heavy use of templates, since GPU kernels in the traversals require compile-time knowledge
 * of the container, traversal types and force functors.
 */
class Simulation {
   private:
    const SimulationParams params;
    const Vec3 box_min;
    const Vec3 box_max;

   public:
    Simulation(const SimulationParams& params)
        : params(params), box_min{0.0, 0.0, 0.0}, box_max(params.domain_size) {}

    /**
     * Runs the simulation with the configured algorithm and functor.
     */
    void run() {
        runWithForceFunctor([this](const auto& force) { runAlgorithm(force); });
    }

    /**
     * Returns a string listing the algorithms that are compiled into this build.
     */
    static std::string availableAlgorithms() {
#if defined(LC_BACKEND_KOKKOS)
        return "cpu | kokkos-soa | kokkos-soa-newton3 | kokkos-aos "
               "| kokkos-c01-soa | kokkos-c01-aos | kokkos-c01-scratch-soa "
               "| kokkos-c01-scratch-aos | kokkos-c08-soa "
               "| kokkos-c08-scratch-soa";
#elif defined(LC_BACKEND_CUDA)
        return "cuda-soa | cuda-aos";
#elif defined(LC_BACKEND_SYCL)
        return "sycl-soa | sycl-aos";
#endif
    }

   private:
    /**
     * Builds the force functor and passes it to the provided function.
     */
    template <typename Fn>
    void runWithForceFunctor(Fn&& fn) {
        switch (params.functor_type) {
            case FunctorType::LENNARD_JONES:
                fn(LennardJonesFunctor(params.epsilon, params.sigma));
                break;
            case FunctorType::HIGH_COMPUTE:
                fn(HighComputeFunctor());
                break;
            case FunctorType::LOW_COMPUTE:
                fn(LowComputeFunctor());
                break;
            case FunctorType::FULL_PARTICLE:
                fn(FullParticleFunctor());
                break;
        }
    }

    /**
     * Selects the container/traversal pair for the configured algorithm
     * and runs the simulation with the provided force functor.
     */
    template <typename ForceT>
    void runAlgorithm(const ForceT& force) {
        switch (params.algorithm_type) {
#if defined(LC_BACKEND_KOKKOS)
            case AlgorithmType::CPU:
                runWith<CpuAoSContainer, CpuTraversal>(force);
                break;
            case AlgorithmType::KOKKOS_SOA:
                runWith<KokkosSoAContainer, KokkosSoATraversal>(force);
                break;
            case AlgorithmType::KOKKOS_SOA_NEWTON3:
                runWith<KokkosSoAContainer, KokkosSoANewton3Traversal>(force);
                break;
            case AlgorithmType::KOKKOS_AOS:
                runWith<KokkosAoSContainer, KokkosAoSTraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C01_SOA:
                runWith<KokkosSoAContainer, KokkosC01SoATraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C01_AOS:
                runWith<KokkosAoSContainer, KokkosC01AoSTraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C01_SCRATCH_SOA:
                runWith<KokkosSoAContainer, KokkosC01ScratchSoATraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C01_SCRATCH_AOS:
                runWith<KokkosAoSContainer, KokkosC01ScratchAoSTraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C08_SOA:
                runWith<KokkosSoAContainer, KokkosC08SoATraversal>(force);
                break;
            case AlgorithmType::KOKKOS_C08_SCRATCH_SOA:
                runWith<KokkosSoAContainer, KokkosC08ScratchSoATraversal>(force);
                break;
#elif defined(LC_BACKEND_CUDA)
            case AlgorithmType::CUDA_SOA:
                runWith<CudaSoAContainer, CudaSoATraversal>(force);
                break;
            case AlgorithmType::CUDA_AOS:
                runWith<CudaAoSContainer, CudaAoSTraversal>(force);
                break;
#elif defined(LC_BACKEND_SYCL)
            case AlgorithmType::SYCL_SOA:
                runWith<SyclSoAContainer, SyclSoATraversal>(force);
                break;
            case AlgorithmType::SYCL_AOS:
                runWith<SyclAoSContainer, SyclAoSTraversal>(force);
                break;
#endif
            default:
                throw std::invalid_argument(
                    "algorithm '" + getKeyFromAlgorithmType(params.algorithm_type) +
                    "' is not compiled into this build; it accepts " + availableAlgorithms() +
                    ". Configure a build with the matching LC_BACKEND.");
        }
    }

    /**
     * Runs the selected loop with the concrete container/traversal pair, timing
     * each iteration, then reports from the profiler.
     */
    template <typename Container, typename Traversal, typename ForceT>
    void runWith(const ForceT& force) {
        Container container(box_min, box_max, params.interaction_radius);
        Traversal traversal(&container);
        Profiler profiler;

        seedParticles(container);

        if (params.only_forces) {
            /**
             * Run a timing-only loop with multiple iterations. No time integration or VTK output is performed.
             * Each iteration has the following steps:
             * 1. Rebuild container for moved particles
             * 2. Compute the pairwise forces
             */
            std::cout << "Run force and rebuild timing..." << std::endl;
            for (float_type curr_t = 0; curr_t < params.end_t; curr_t += params.delta_t) {
                profiler.beginSample();
                container.rebuild();
                profiler.section("rebuild");
                traversal.applyPairwise(force);
                profiler.section("applyPairwise");
                profiler.endSample();
            }
        } else {
            /**
             * Run a full simulation with multiple iterations and timing measurements.
             * Each iteration has multiple steps:
             * 1. Clear previous forces
             * 2. Rebuild container for moved particles
             * 3. Compute the pairwise forces
             * 4. Time integration with reflecting boundaries
             * 5. (If enabled) Write a VTK snapshot.
             */
            std::cout << "Running full simulation..." << std::endl;

            std::optional<VTKWriter> writer;
            if (params.output_vtk_files) {
                writer.emplace(params.output_dir / "vtk_output", runTag());
                writer->writeCellGrid(container.getNumCellsWithHalo(), container.getCellSize(),
                                    container.getBoxMinWithHalo(), container.getCellIndexing());
                writer->write(0, container.getParticles());
            }

            const TimestepFunctor step{params.delta_t, box_min, box_max};
            const ResetForceFunctor reset{};
            size_t timestep = 0;
            for (float_type curr_t = 0; curr_t < params.end_t; curr_t += params.delta_t) {
                ++timestep;
                std::cout << "Timestep " << timestep << " (t = " << curr_t << ")" << std::endl;
                profiler.beginSample();
                traversal.applyForEach(reset);
                profiler.section("applyForEach_forceReset");
                container.rebuild();
                profiler.section("rebuild");
                traversal.applyPairwise(force);
                profiler.section("applyPairwise");
                traversal.applyForEach(step);
                profiler.section("applyForEach_integration");
                profiler.endSample();
                // Outside the sample: the host copy and file write are not timestep work.
                if (writer) writer->write(timestep, container.getParticles());
            }
        }

        printResults(profiler);
        if (!params.output_dir.empty()) {
            writeProfileToFile(profiler);
            writeSummaryToFile(profiler);
        }
    }

    // --- Particle initialization ---
    
    void seedParticles(auto& container) const {
        std::cout << "Adding particles..." << std::endl;
        uint_type next_id = 0;
        for (const auto& initializer : params.initializers) {
            next_id = std::visit(
                [&](const auto& init_params) { return seedWith(container, init_params, next_id); },
                initializer);
            }
        }
        
        static uint_type seedWith(auto& container, const RandomInitializerParams& init_params,
            uint_type first_id) {
                return RandomInitializer(init_params).initialize(container, first_id);
            }
            
            static uint_type seedWith(auto& container, const CuboidInitializerParams& init_params,
                uint_type first_id) {
                    return CuboidInitializer(init_params).initialize(container, first_id);
                }
                
    // --- Result printing ---

    // Algorithm key determines the backend, SYCL inserts also whether
    // the target is generic of CUDA
    std::string runTag() const {
        std::string algorithm = getKeyFromAlgorithmType(params.algorithm_type);
        #ifdef LC_SYCL_FLOW
        algorithm.insert(4, "-" LC_SYCL_FLOW);  // 4 == length of the "sycl" prefix
        #endif
        return algorithm + "_" + getKeyFromFunctorType(params.functor_type);
    }
                
    // Human-readable timing report block: header line plus the profiler summary table.
    std::string timingReport(const Profiler& profiler) const {
        std::ostringstream out;
        out << "--- Per-timestep Traversal Timing (over " << profiler.numSamples()
            << " iterations) ---\n\n"
            << profiler.summary(runTag());
        return out.str();
    }

    void printResults(const Profiler& profiler) const {
        std::cout << "\n\n" << timingReport(profiler);
    }

    void writeProfileToFile(const Profiler& profiler) const {
        const auto path = params.output_dir / "timings.csv";
        std::ofstream file(path);
        if (!file) {
            std::cerr << "Error: could not open output file '" << path.string() << "'.\n";
            return;
        }
        profiler.writeCsv(file);
    }

    void writeSummaryToFile(const Profiler& profiler) const {
        const auto path = params.output_dir / "summary.txt";
        std::ofstream file(path);
        if (!file) {
            std::cerr << "Error: could not open output file '" << path.string() << "'.\n";
            return;
        }
        file << params.toString() << timingReport(profiler);
    }
};
