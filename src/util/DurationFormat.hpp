#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>

// Scale a nanosecond duration to the largest unit that keeps the value >= 1,
// returning the scaled value and its unit string (e.g. {1.5, "ms"}).
inline std::pair<double, std::string> formatDuration(std::chrono::nanoseconds duration) {
    const double nanoseconds = static_cast<double>(duration.count());
    const double absoluteNanoseconds = std::fabs(nanoseconds);

    if (absoluteNanoseconds < 1'000.0) {
        return {nanoseconds, "ns"};
    }
    if (absoluteNanoseconds < 1'000'000.0) {
        return {nanoseconds / 1'000.0, "us"};
    }
    if (absoluteNanoseconds < 1'000'000'000.0) {
        return {nanoseconds / 1'000'000.0, "ms"};
    }
    if (absoluteNanoseconds < 60.0 * 1'000'000'000.0) {
        return {nanoseconds / 1'000'000'000.0, "s"};
    }
    if (absoluteNanoseconds < 60.0 * 60.0 * 1'000'000'000.0) {
        return {nanoseconds / (60.0 * 1'000'000'000.0), "min"};
    }
    return {nanoseconds / (60.0 * 60.0 * 1'000'000'000.0), "h"};
}
