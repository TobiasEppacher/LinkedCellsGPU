#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

enum class AlgorithmType {
    CPU,
    KOKKOS_SOA,
    KOKKOS_SOA_NEWTON3,
    KOKKOS_AOS,
    KOKKOS_C01_SOA,
    KOKKOS_C01_AOS,
    KOKKOS_C01_SCRATCH_SOA,
    KOKKOS_C01_SCRATCH_AOS,
    KOKKOS_C08_SOA,
    KOKKOS_C08_SCRATCH_SOA,
    CUDA_SOA,
    CUDA_AOS,
    SYCL_SOA,
    SYCL_AOS
};

inline const std::map<std::string, AlgorithmType> ALGORITHM_TYPE_KEY_MAP = {
    {"cpu", AlgorithmType::CPU},
    {"kokkos-soa", AlgorithmType::KOKKOS_SOA},
    // Same as 'kokkos-soa', but each pair is evaluated once and atomically added to both particles.
    {"kokkos-soa-newton3", AlgorithmType::KOKKOS_SOA_NEWTON3},
    // Same traversal as 'kokkos-soa', but on a container storing Particle structs
    // instead of parallel scalar arrays.
    {"kokkos-aos", AlgorithmType::KOKKOS_AOS},
    {"kokkos-c01-soa", AlgorithmType::KOKKOS_C01_SOA},
    // Same traversal as 'kokkos-c01-soa', but on a container storing Particle structs
    // instead of parallel scalar arrays.
    {"kokkos-c01-aos", AlgorithmType::KOKKOS_C01_AOS},
    // Same cell-per-team decomposition as 'kokkos-c01-soa', but staging the neighbourhood
    // in team scratch instead of reading positions from global memory.
    {"kokkos-c01-scratch-soa", AlgorithmType::KOKKOS_C01_SCRATCH_SOA},
    // Same as 'kokkos-c01-scratch-soa', but staging Particle structs instead of
    // parallel scalar arrays, in both global memory and scratch.
    {"kokkos-c01-scratch-aos", AlgorithmType::KOKKOS_C01_SCRATCH_AOS},
    {"kokkos-c08-soa", AlgorithmType::KOKKOS_C08_SOA},
    // Same base-cell-per-team decomposition as 'kokkos-c08-soa', but staging the
    // 2x2x2 base block in team scratch instead of reading it from global memory.
    {"kokkos-c08-scratch-soa", AlgorithmType::KOKKOS_C08_SCRATCH_SOA},
    {"cuda-soa", AlgorithmType::CUDA_SOA},
    // Same traversal as 'cuda-soa', but on a container storing Particle structs
    // instead of parallel scalar arrays.
    {"cuda-aos", AlgorithmType::CUDA_AOS},
    {"sycl-soa", AlgorithmType::SYCL_SOA},
    // Same traversal as 'sycl-soa', but on a container storing Particle structs
    // instead of parallel scalar arrays.
    {"sycl-aos", AlgorithmType::SYCL_AOS}
};

inline AlgorithmType getAlgorithmTypeFromKey(const std::string& key) {
    auto it = ALGORITHM_TYPE_KEY_MAP.find(key);
    if (it != ALGORITHM_TYPE_KEY_MAP.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Unknown algorithm key: " + key);
    }
}

inline std::string getKeyFromAlgorithmType(AlgorithmType type) {
    for (const auto& pair : ALGORITHM_TYPE_KEY_MAP) {
        if (pair.second == type) {
            return pair.first;
        }
    }
    throw std::invalid_argument("Unknown algorithm type");
}

inline std::vector<std::string> getAlgorithmKeyList() {
    std::vector<std::string> keys;
    for (const auto& pair : ALGORITHM_TYPE_KEY_MAP) {
        keys.push_back(pair.first);
    }
    return keys;
}
