#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "ParticleFixture.hpp"
#include "container/CpuAoSContainer.hpp"
#include "container/KokkosAoSContainer.hpp"
#include "container/KokkosSoAContainer.hpp"
#include "functors/pairwise_force/LennardJonesFunctor.hpp"
#include "functors/unary/TimestepFunctor.hpp"
#include "traversal/CpuTraversal.hpp"
#include "traversal/KokkosSoATraversal.hpp"
#include "traversal/KokkosSoANewton3Traversal.hpp"
#include "traversal/KokkosAoSTraversal.hpp"
#include "traversal/KokkosC01AoSTraversal.hpp"
#include "traversal/KokkosC01ScratchAoSTraversal.hpp"
#include "traversal/KokkosC01ScratchSoATraversal.hpp"
#include "traversal/KokkosC01SoATraversal.hpp"
#include "traversal/KokkosC08ScratchSoATraversal.hpp"
#include "traversal/KokkosC08SoATraversal.hpp"
#include "util/CompareUtils.hpp"

/**
 * Runs NUM_TIMESTEPS interlaced force/integration/reset steps once per traversal and
 * compares the result against an independent O(N^2) reference, so both the force
 * computation and the integration/boundary kernel are checked over many steps.
 * cuda_force_test and sycl_force_test pin their traversals to the same reference,
 * which is what ties the three backends to each other.
 */

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int exit_code = 0;
    {
        using namespace test_fixture;

        const LennardJonesFunctor lj{LJ_EPSILON, LJ_SIGMA};

        CpuAoSContainer cpu_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_newton3_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosAoSContainer kokkos_aos_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_c01_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosAoSContainer kokkos_c01_aos_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_c01_scratch_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosAoSContainer kokkos_c01_scratch_aos_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_c08_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        KokkosSoAContainer kokkos_c08_scratch_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);

        const std::vector<Particle> particles = buildParticles();

        for (const auto& p : particles) {
            cpu_container.addParticle(p);
            kokkos_container.addParticle(p);
            kokkos_newton3_container.addParticle(p);
            kokkos_aos_container.addParticle(p);
            kokkos_c01_container.addParticle(p);
            kokkos_c01_aos_container.addParticle(p);
            kokkos_c01_scratch_container.addParticle(p);
            kokkos_c01_scratch_aos_container.addParticle(p);
            kokkos_c08_container.addParticle(p);
            kokkos_c08_scratch_container.addParticle(p);
        }

        const TimestepFunctor step{DT, BOX_MIN, BOX_MAX};
        const ResetForceFunctor reset{};

        CpuTraversal cpu_traversal(&cpu_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            cpu_container.rebuild();
            cpu_traversal.applyPairwise(lj);
            cpu_traversal.applyForEach(step);
            cpu_traversal.applyForEach(reset);
        }

        KokkosSoATraversal kokkos_traversal(&kokkos_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_container.rebuild();
            kokkos_traversal.applyPairwise(lj);
            kokkos_traversal.applyForEach(step);
            kokkos_traversal.applyForEach(reset);
        }

        KokkosSoANewton3Traversal kokkos_newton3_traversal(&kokkos_newton3_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_newton3_container.rebuild();
            kokkos_newton3_traversal.applyPairwise(lj);
            kokkos_newton3_traversal.applyForEach(step);
            kokkos_newton3_traversal.applyForEach(reset);
        }

        KokkosAoSTraversal kokkos_aos_traversal(&kokkos_aos_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_aos_container.rebuild();
            kokkos_aos_traversal.applyPairwise(lj);
            kokkos_aos_traversal.applyForEach(step);
            kokkos_aos_traversal.applyForEach(reset);
        }

        KokkosC01SoATraversal kokkos_c01_traversal(&kokkos_c01_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c01_container.rebuild();
            kokkos_c01_traversal.applyPairwise(lj);
            kokkos_c01_traversal.applyForEach(step);
            kokkos_c01_traversal.applyForEach(reset);
        }

        KokkosC01AoSTraversal kokkos_c01_aos_traversal(&kokkos_c01_aos_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c01_aos_container.rebuild();
            kokkos_c01_aos_traversal.applyPairwise(lj);
            kokkos_c01_aos_traversal.applyForEach(step);
            kokkos_c01_aos_traversal.applyForEach(reset);
        }

        KokkosC01ScratchSoATraversal kokkos_c01_scratch_traversal(&kokkos_c01_scratch_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c01_scratch_container.rebuild();
            kokkos_c01_scratch_traversal.applyPairwise(lj);
            kokkos_c01_scratch_traversal.applyForEach(step);
            kokkos_c01_scratch_traversal.applyForEach(reset);
        }

        KokkosC01ScratchAoSTraversal kokkos_c01_scratch_aos_traversal(&kokkos_c01_scratch_aos_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c01_scratch_aos_container.rebuild();
            kokkos_c01_scratch_aos_traversal.applyPairwise(lj);
            kokkos_c01_scratch_aos_traversal.applyForEach(step);
            kokkos_c01_scratch_aos_traversal.applyForEach(reset);
        }

        KokkosC08SoATraversal kokkos_c08_traversal(&kokkos_c08_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c08_container.rebuild();
            kokkos_c08_traversal.applyPairwise(lj);
            kokkos_c08_traversal.applyForEach(step);
            kokkos_c08_traversal.applyForEach(reset);
        }

        KokkosC08ScratchSoATraversal kokkos_c08_scratch_traversal(&kokkos_c08_scratch_container);
        for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
            kokkos_c08_scratch_container.rebuild();
            kokkos_c08_scratch_traversal.applyPairwise(lj);
            kokkos_c08_scratch_traversal.applyForEach(step);
            kokkos_c08_scratch_traversal.applyForEach(reset);
        }

        const std::vector<Particle> reference = runReference(particles, lj, step, reset);

        // Compare final positions, velocities and forces against the reference, matched by id.
        bool all_ok = true;
        all_ok &= compareParticles(cpu_container.getParticles(), reference, "CPU", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_container.getParticles(), reference, "Kokkos", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_newton3_container.getParticles(), reference, "Kokkos Newton3", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_aos_container.getParticles(), reference, "Kokkos AoS", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c01_container.getParticles(), reference, "Kokkos C01", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c01_aos_container.getParticles(), reference, "Kokkos C01 AoS", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c01_scratch_container.getParticles(), reference, "Kokkos C01 scratch", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c01_scratch_aos_container.getParticles(), reference, "Kokkos C01 scratch AoS", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c08_container.getParticles(), reference, "Kokkos C08", "Reference", FLOAT_EPS);
        all_ok &= compareParticles(kokkos_c08_scratch_container.getParticles(), reference, "Kokkos C08 scratch", "Reference", FLOAT_EPS);

        // Invariants the traversals rely on: no particle ever reaches the halo ring.
        CpuAoSContainer rejection_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
        all_ok &= checkBoundaryStepStaysInside();
        all_ok &= checkRejectsOutOfDomain(rejection_container);

        std::cout << "\n--- Multi-step Simulation Test (" << NUM_TIMESTEPS << " interlaced timesteps) ---\n";
        std::cout << (all_ok ? "PASS: Positions, velocities and forces after the run are within tolerance.\n"
                             : "FAIL: At least one traversal differs from the reference.\n");

        exit_code = all_ok ? 0 : 1;
    }
    Kokkos::finalize();
    return exit_code;
}
