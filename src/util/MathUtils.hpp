#pragma once

#include "DeviceDefinitions.hpp"
#include "TypeDefinitions.hpp"

LC_HOST_DEVICE_INLINE
float_type dist(const Vec3& r1, const Vec3& r2) {
    const float_type x = r1[0] - r2[0];
    const float_type y = r1[1] - r2[1];
    const float_type z = r1[2] - r2[2];
    return lc::sqrt(x*x + y*y + z*z);
}

LC_HOST_DEVICE_INLINE
float_type dist_sq(const Vec3& r1, const Vec3& r2) {
    const float_type x = r1[0] - r2[0];
    const float_type y = r1[1] - r2[1];
    const float_type z = r1[2] - r2[2];
    return x*x + y*y + z*z;
}
