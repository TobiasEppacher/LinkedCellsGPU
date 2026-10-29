#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "../particle/Particle.hpp"
#include "TypeDefinitions.hpp"

// Mixed absolute/relative tolerance so the check works for both tiny and large magnitudes.
inline bool approxEqual(double a, double b, double eps) {
    double diff = std::fabs(a - b);
    return diff <= eps || diff <= eps * std::max(std::fabs(a), std::fabs(b));
}

/**
 * Compares two particle sets field by field, matching them by id. Returns true
 * if they agree within eps and prints every mismatch otherwise. a_name/b_name
 * label the two sides in the diagnostic output so callers can compare any pair
 * of traversals (CPU vs Kokkos, Kokkos vs Kokkos, ...), not just a fixed pair.
 */
inline bool compareParticles(const std::vector<Particle>& a_particles,
                             const std::vector<Particle>& b_particles,
                             const std::string& a_name = "A",
                             const std::string& b_name = "B",
                             double eps = 1e-9) {
    bool equal = true;

    // Check particle counts
    if (a_particles.size() != b_particles.size()) {
        std::cout << "Particle count mismatch: " << a_name << "=" << a_particles.size()
                  << " " << b_name << "=" << b_particles.size() << std::endl;
        equal = false;
    }

    // Check for duplicate ids
    std::unordered_map<uint_type, const Particle*> a_by_id;
    a_by_id.reserve(a_particles.size());
    for (const auto& p : a_particles) {
        if (!a_by_id.emplace(p.id, &p).second) {
            std::cout << "Duplicate id " << p.id << " in " << a_name << std::endl;
            equal = false;
        }
    }
    std::unordered_map<uint_type, const Particle*> b_by_id;
    b_by_id.reserve(b_particles.size());
    for (const auto& p : b_particles) {
        if (!b_by_id.emplace(p.id, &p).second) {
            std::cout << "Duplicate id " << p.id << " in " << b_name << std::endl;
            equal = false;
        }
    }

    // Check particle equality
    for (const auto& a : a_particles) {
        auto it = b_by_id.find(a.id);
        if (it == b_by_id.end()) {
            std::cout << "id " << a.id << " present in " << a_name
                      << " but missing in " << b_name << std::endl;
            equal = false;
            continue;
        }
        const Particle& b = *it->second;

        const char* names[3] = {"r", "v", "f"};
        const Vec3* a_fields[3] = {&a.r, &a.v, &a.f};
        const Vec3* b_fields[3] = {&b.r, &b.v, &b.f};
        for (int field = 0; field < 3; ++field) {
            for (int d = 0; d < 3; ++d) {
                double av = (*a_fields[field])[d];
                double bv = (*b_fields[field])[d];
                if (!approxEqual(av, bv, eps)) {
                    std::cout << "id " << a.id << " " << names[field] << "[" << d
                              << "] mismatch: " << a_name << "=" << av << " "
                              << b_name << "=" << bv << std::endl;
                    equal = false;
                }
            }
        }
    }

    return equal;
}
