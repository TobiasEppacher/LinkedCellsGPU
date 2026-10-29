#include <cstring>
#include <filesystem>
#include <iostream>

#include "io/ConfigFileLoader.hpp"
#include "io/SimulationParams.hpp"
#include "simulation/Simulation.hpp"

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml>\n"
              << "  The YAML config file is the only input; every key below is mandatory\n"
              << "  except 'output_dir', and unknown keys are rejected.\n\n"
              << "  delta_t: 0.01\n"
              << "  end_t: 1.0                       # iterations run until the time passes end_t\n"
              << "  domain_size: [10.0, 10.0, 10.0]  # domain is [0, 0, 0] to domain_size\n"
              << "  interaction_radius: 1.0\n"
              << "  epsilon: 1.0                     # Lennard-Jones well depth\n"
              << "  sigma: 0.4                       # Lennard-Jones zero-crossing distance\n"
              << "  algorithm: ...                   # this build accepts:\n"
              << "                                   #   " << Simulation::availableAlgorithms() << "\n"
              << "  functor: lj                      # lj | high | low | full\n"
              << "  only_forces: false               # true repeats the force calculation only\n"
              << "  output_vtk_files: true           # per-timestep snapshots; needs output_dir\n"
              << "  output_dir: \"./output\"           # optional; profiling CSV and summary go here\n"
              << "  initializers:                    # applied in order\n"
              << "    - type: random\n"
              << "      num_particles: 1000\n"
              << "      box_min: [0.0, 0.0, 0.0]\n"
              << "      box_max: [10.0, 10.0, 10.0]\n"
              << "    - type: cuboid\n"
              << "      box_min: [0.0, 0.0, 0.0]\n"
              << "      box_max: [5.0, 5.0, 5.0]\n"
              << "      grid_counts: [10, 10, 10]\n";
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    if (argc != 2) {
        std::cerr << "Error: exactly one argument is expected, the config file path.\n";
        printUsage(argv[0]);
        return 1;
    }

    SimulationParams params;
    try {
        params = ConfigFileLoader::load(argv[1]);
    } catch (const ConfigError& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Guards against a loop that never terminates; the loader only checks types.
    if (params.delta_t <= 0) {
        std::cerr << "Error: 'delta_t' must be greater than zero.\n";
        return 1;
    }
    if (params.epsilon <= 0 || params.sigma <= 0) {
        std::cerr << "Error: 'epsilon' and 'sigma' must be greater than zero.\n";
        return 1;
    }
    if (params.output_vtk_files && params.output_dir.empty()) {
        std::cerr << "Error: 'output_vtk_files' requires an 'output_dir'.\n";
        return 1;
    }

    // Refuse to reuse an existing output directory so earlier runs' data is never overwritten.
    if (!params.output_dir.empty()) {
        if (std::filesystem::exists(params.output_dir)) {
            std::cerr << "Error: output directory '" << params.output_dir.string()
                      << "' already exists; refusing to overwrite existing data. "
                         "Choose a new output directory.\n";
            return 1;
        }
        std::filesystem::create_directories(params.output_dir);
        // Keep the exact input alongside the results for reproducibility.
        std::filesystem::copy_file(params.input_file_path, params.output_dir / "config.yaml");
    }

    std::cout << params.toString();

#if defined(LC_BACKEND_KOKKOS)
    Kokkos::initialize();
#endif
    int exit_code = 0;
    try {
        Simulation simulation(params);
        simulation.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        exit_code = 1;
    }

#if defined(LC_BACKEND_KOKKOS)
    Kokkos::finalize();
#endif

    return exit_code;
}
