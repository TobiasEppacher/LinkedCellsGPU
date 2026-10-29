#include <iostream>
#include <vector>

#include "ParticleFixture.hpp"
#include "container/SyclAoSContainer.hpp"
#include "container/SyclSoAContainer.hpp"
#include "functors/pairwise_force/LennardJonesFunctor.hpp"
#include "functors/unary/TimestepFunctor.hpp"
#include "traversal/SyclAoSTraversal.hpp"
#include "traversal/SyclSoATraversal.hpp"
#include "util/CompareUtils.hpp"

/**
 * Runs NUM_TIMESTEPS interlaced force/integration/reset steps through SyclSoATraversal and
 * compares against the O(N^2) reference from ParticleFixture.hpp. kokkos_force_test and
 * cuda_force_test pin their traversals to that same reference, so agreeing with it pins
 * every backend to the others. A separate binary because acpp compiles this build.
 */

int main() {
    using namespace test_fixture;

    const LennardJonesFunctor lj{LJ_EPSILON, LJ_SIGMA};
    const TimestepFunctor step{DT, BOX_MIN, BOX_MAX};
    const ResetForceFunctor reset{};

    const std::vector<Particle> particles = buildParticles();

    SyclSoAContainer container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
    for (const auto& p : particles) container.addParticle(p);

    SyclSoATraversal traversal(&container);
    for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
        container.rebuild();
        traversal.applyPairwise(lj);
        traversal.applyForEach(step);
        traversal.applyForEach(reset);
    }

    SyclAoSContainer aos_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
    for (const auto& p : particles) aos_container.addParticle(p);

    SyclAoSTraversal aos_traversal(&aos_container);
    for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
        aos_container.rebuild();
        aos_traversal.applyPairwise(lj);
        aos_traversal.applyForEach(step);
        aos_traversal.applyForEach(reset);
    }

    const std::vector<Particle> reference = runReference(particles, lj, step, reset);

    SyclSoAContainer rejection_container(BOX_MIN, BOX_MAX, INTERACTION_RADIUS);
    bool ok = compareParticles(container.getParticles(), reference, "SYCL SoA", "Reference", FLOAT_EPS);
    ok &= compareParticles(aos_container.getParticles(), reference, "SYCL AoS", "Reference", FLOAT_EPS);
    ok &= checkBoundaryStepStaysInside();
    ok &= checkRejectsOutOfDomain(rejection_container);

    std::cout << "\n--- Multi-step SYCL Simulation Test (" << NUM_TIMESTEPS
              << " interlaced timesteps) ---\n";
    std::cout << (ok ? "PASS: Positions, velocities and forces after the run are within tolerance.\n"
                     : "FAIL: The SYCL traversal differs from the reference.\n");
    return ok ? 0 : 1;
}
