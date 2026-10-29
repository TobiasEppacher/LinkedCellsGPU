#pragma once

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "functors/unary/TimestepFunctor.hpp"
#include "particle/Particle.hpp"
#include "util/MathUtils.hpp"
#include "util/TypeDefinitions.hpp"

// Shared by kokkos_force_test, cuda_force_test and sycl_force_test, which are
// separate binaries per backend, so all three exercise exactly the same particles.
namespace test_fixture {

inline constexpr Vec3 BOX_MIN = {0.0, 0.0, 0.0};
inline constexpr Vec3 BOX_MAX = {10.0, 10.0, 10.0};
// Cells are 1.0 wide -> a particle's cell is floor(r + 1)
inline constexpr float_type INTERACTION_RADIUS = 1.0;

// Sigma puts the LJ equilibrium distance (2^(1/6) * sigma ~ 0.22) inside the separations built below.
inline constexpr float_type LJ_EPSILON = float_type(0.005);
inline constexpr float_type LJ_SIGMA = float_type(0.2);

inline constexpr size_t NUM_TIMESTEPS = 100;
inline constexpr float_type DT = float_type(0.05);

// Covers every stencil offset, self- and cross-cell clustering, cutoff exclusion,
// isolated particles, particles hugging the domain faces, and boundary reflection.
inline std::vector<Particle> buildParticles() {
    std::vector<Particle> particles;
    uint_type next_id = 0;
    auto add = [&](float_type x, float_type y, float_type z, float_type vx = float_type(0),
                   float_type vy = float_type(0), float_type vz = float_type(0)) {
        Particle p;
        p.id = next_id++;
        p.r = {x, y, z};
        p.v = {vx, vy, vz};
        particles.push_back(p);
    };

    // Two particles 0.2 apart across a cell boundary along `dir` -> adjacent cells.
    auto add_cross_pair = [&](Vec3 com, std::array<int, 3> dir) {
        Vec3 a, b;
        for (int d = 0; d < 3; ++d) {
            a[d] = com[d] - float_type(0.1) * dir[d];
            b[d] = com[d] + float_type(0.1) * dir[d];
        }
        add(a[0], a[1], a[2]);
        add(b[0], b[1], b[2]);
    };
    // Three particles inside one cell -> mutual self-cell interaction.
    auto add_self_cluster = [&](Vec3 anchor) {
        const float_type cx = anchor[0] + float_type(0.5);
        const float_type cy = anchor[1] + float_type(0.5);
        const float_type cz = anchor[2] + float_type(0.5);
        add(cx - float_type(0.15), cy, cz);
        add(cx + float_type(0.15), cy, cz);
        add(cx, cy + float_type(0.15), cz);
    };
    // Two particles per cell across a boundary, all four inside the cutoff.
    auto add_multi_cross = [&](Vec3 anchor) {
        const float_type bx = anchor[0];  // integer cell boundary along x
        const float_type y = anchor[1] + float_type(0.5);
        const float_type z = anchor[2] + float_type(0.5);
        add(bx - float_type(0.1), y - float_type(0.15), z);
        add(bx - float_type(0.1), y + float_type(0.15), z);
        add(bx + float_type(0.1), y - float_type(0.15), z);
        add(bx + float_type(0.1), y + float_type(0.15), z);
    };
    // 1.5 apart with a full empty cell between them -> the cutoff must exclude them.
    auto add_gap_pair = [&](Vec3 anchor) {
        const float_type cy = anchor[1] + float_type(0.5);
        const float_type cz = anchor[2] + float_type(0.5);
        add(anchor[0] - float_type(0.25), cy, cz);
        add(anchor[0] + float_type(1.25), cy, cz);
    };
    // No neighbour inside the cutoff -> force stays zero and it never moves.
    auto add_isolated = [&](Vec3 anchor) {
        add(anchor[0] + float_type(0.5), anchor[1] + float_type(0.5), anchor[2] + float_type(0.5));
    };

    // Anchors spaced 3 cells apart, so the cases never interact with one another.
    std::vector<Vec3> sites;
    const float_type S[3] = {float_type(2), float_type(5), float_type(8)};
    for (float_type sx : S)
        for (float_type sy : S)
            for (float_type sz : S) sites.push_back({sx, sy, sz});
    size_t next_site = 0;

    // One cross-cell pair per unique neighbour direction, covering every stencil offset.
    const std::array<std::array<int, 3>, 13> directions = {{
        {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},
        {{1, 1, 0}}, {{1, -1, 0}}, {{1, 0, 1}}, {{1, 0, -1}},
        {{0, 1, 1}}, {{0, 1, -1}},
        {{1, 1, 1}}, {{1, 1, -1}}, {{1, -1, 1}}, {{1, -1, -1}},
    }};
    for (const auto& dir : directions) {
        const Vec3 anchor = sites[next_site++];
        Vec3 com;
        for (int d = 0; d < 3; ++d)
            com[d] = (dir[d] != 0) ? anchor[d] : anchor[d] + float_type(0.5);
        add_cross_pair(com, dir);
    }

    // 27 anchors total: 13 cross pairs + 4 self clusters + 3 multi-cross + 4 gaps + 3 lone.
    add_self_cluster(sites[next_site++]);
    add_self_cluster(sites[next_site++]);
    add_self_cluster(sites[next_site++]);
    add_self_cluster(sites[next_site++]);
    add_multi_cross(sites[next_site++]);
    add_multi_cross(sites[next_site++]);
    add_multi_cross(sites[next_site++]);
    add_gap_pair(sites[next_site++]);
    add_gap_pair(sites[next_site++]);
    add_gap_pair(sites[next_site++]);
    add_gap_pair(sites[next_site++]);
    add_isolated(sites[next_site++]);
    add_isolated(sites[next_site++]);
    add_isolated(sites[next_site++]);

    /**
     * Pairs hugging a domain face, the outer partner moving outward so it reflects on
     * the first step: box_min is the first interior cell, box_max belongs to the halo,
     * and the last interior cell's stencil reaches into it. Kept away from the sites
     * above so each pair only interacts with itself.
     */
    const float_type hug = float_type(0.001), drift = float_type(0.2);
    add(BOX_MAX[0] - hug, float_type(3.5), float_type(3.5), drift);
    add(BOX_MAX[0] - float_type(0.2), float_type(3.5), float_type(3.5));
    add(BOX_MIN[0], float_type(6.5), float_type(3.5), -drift);
    add(BOX_MIN[0] + float_type(0.2), float_type(6.5), float_type(3.5));
    add(float_type(3.5), BOX_MAX[1] - hug, float_type(6.5), float_type(0), drift);
    add(float_type(3.5), BOX_MAX[1] - float_type(0.2), float_type(6.5));

    // One per domain corner, moving outward so it reflects; also covers the corner cells
    // whose stencil reaches into the halo.
    const float_type low = float_type(0.6), high = float_type(9.4);
    const float_type speed = float_type(0.25);
    for (int cx = 0; cx < 2; ++cx)
        for (int cy = 0; cy < 2; ++cy)
            for (int cz = 0; cz < 2; ++cz)
                add(cx ? high : low, cy ? high : low, cz ? high : low, cx ? speed : -speed,
                    cy ? speed : -speed, cz ? speed : -speed);

    return particles;
}

/**
 * No integration step may end on box_max: that position bins into the halo cell, which
 * the traversals never visit and whose stencil leaves the grid. Reflection maps box_max
 * onto itself, so the functor has to push it inside. Checked over several domain sizes
 * because that push must outlive the float rounding at box_max (the series scripts run
 * domains up to 64 cells).
 */
inline bool checkBoundaryStepStaysInside() {
    bool ok = true;
    for (float_type extent : {BOX_MAX[0], float_type(64), float_type(128), float_type(1024)}) {
        const Vec3 box_min{float_type(0), float_type(0), float_type(0)};
        const Vec3 box_max{extent, extent, extent};
        const TimestepFunctor step{DT, box_min, box_max};
        for (int d = 0; d < 3; ++d) {
            for (int moving = 0; moving < 2; ++moving) {
                Particle p;
                p.r = {extent / 2, extent / 2, extent / 2};
                p.r[d] = extent;
                if (moving) p.v[d] = float_type(1);
                step(p);
                if (p.r[d] < box_min[d] || p.r[d] >= extent) {
                    std::cout << "Step from box_max = " << extent << " on axis " << d
                              << (moving ? " (moving)" : " (at rest)")
                              << " left [box_min, box_max): r = " << p.r[d] << "\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// Seeding outside [box_min, box_max) must be rejected rather than binned into the halo.
template <typename Container>
inline bool checkRejectsOutOfDomain(Container& container) {
    bool ok = true;
    auto rejects = [&](const char* what, const Vec3& r) {
        Particle p;
        p.r = r;
        try {
            container.addParticle(p);
            std::cout << "Accepted an out-of-domain particle (" << what << ")\n";
            ok = false;
        } catch (const std::out_of_range&) {
        }
    };
    rejects("on box_max", BOX_MAX);
    rejects("below box_min", {BOX_MIN[0] - float_type(0.5), float_type(5), float_type(5)});
    rejects("above box_max", {float_type(5), BOX_MAX[1] + float_type(0.5), float_type(5)});
    return ok;
}

// Independent brute-force O(N^2) reference over the same NUM_TIMESTEPS.
template <typename ForceT, typename StepT, typename ResetT>
inline std::vector<Particle> runReference(const std::vector<Particle>& initial, const ForceT& force,
                                          const StepT& step, const ResetT& reset) {
    const float_type interaction_radius_sq = INTERACTION_RADIUS * INTERACTION_RADIUS;
    std::vector<Particle> reference = initial;
    for (size_t t = 0; t < NUM_TIMESTEPS; ++t) {
        for (size_t i = 0; i < reference.size(); ++i) {
            for (size_t j = 0; j < reference.size(); ++j) {
                if (i == j || dist_sq(reference[i].r, reference[j].r) > interaction_radius_sq) continue;
                const Vec3 f = force(reference[i], reference[j]);
                reference[i].f[0] += f[0];
                reference[i].f[1] += f[1];
                reference[i].f[2] += f[2];
            }
        }
        for (auto& p : reference) {
            step(p);
            reset(p);
        }
    }
    return reference;
}

}  // namespace test_fixture
