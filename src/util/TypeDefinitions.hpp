#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * Precision used for all particle-related quantities (positions, velocities,
 * forces, distances, ids, ...). Controlled by the CMake option USE_64_BIT
 * (default OFF), which defines LC_USE_64_BIT to 1 or 0:
 *   0 (default) -> float_type = float,  int_type = int32_t, uint_type = uint32_t (32-bit)
 *   1           -> float_type = double, int_type = int64_t, uint_type = uint64_t (64-bit)
 */
#if defined(LC_USE_64_BIT) && LC_USE_64_BIT
using float_type = double;
using int_type = int64_t;
using uint_type = uint64_t;
#define FLOAT_EPS 1e-9
#else
using float_type = float;
using int_type = int32_t;
using uint_type = uint32_t;
#define FLOAT_EPS 1e-5
#endif

using Vec3 = std::array<float_type, 3>;
using Vec3u = std::array<uint_type, 3>;
