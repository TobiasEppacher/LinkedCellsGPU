#pragma once

#include <string>
#include "../util/DeviceDefinitions.hpp"
#include "../util/TypeDefinitions.hpp"

struct Particle {
    uint_type id;
    Vec3 r;
    Vec3 v;
    Vec3 f;

    // Device-callable so particles can be materialised from SoA buffers inside
    // device kernels. id defaults to 0 (callers assign a real id explicitly).
    LC_HOST_DEVICE_INLINE
    Particle() : id(0), r{0.0, 0.0, 0.0}, v{0.0, 0.0, 0.0}, f{0.0, 0.0, 0.0} {}

    std::string str() const {
        return "Particle{id: " + std::to_string(id) + ", r: [" + std::to_string(r[0]) + ", " + std::to_string(r[1]) + ", " + std::to_string(r[2]) + "]}";
    }
};
