#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "DurationFormat.hpp"  // formatDuration(...)

/**
 * Sampling profiler for repeatedly-called routines (e.g. applyPairwise).
 *
 * Profiler keeps ONE sample per call. A "sample" is the ordered list of section
 * durations measured between section() boundaries, plus the call's total. Over
 * many calls this lets us report count / min / max / mean / median / stddev /
 * sum per section AND for the total execution, and dump the raw per-call data.
 *
 * Usage per call:
 *   beginSample();
 *   ...work...; section("phase A");
 *   ...work...; section("phase B");
 *   endSample();
 *
 * Not thread-safe: intended to be driven serially from a single host thread.
 */
class Profiler {
    using Clock = std::chrono::steady_clock;
    using ns = std::chrono::nanoseconds;

    std::vector<std::string> _labels;     // section order, fixed after first sample
    std::vector<std::vector<ns>> _samples;  // _samples[call][section]
    std::vector<ns> _totals;              // _totals[call]

    // Scratch state for the in-progress call.
    Clock::time_point _callStart{};
    Clock::time_point _lastSplit{};
    std::vector<ns> _current;
    std::size_t _cursor = 0;
    bool _inSample = false;

   public:
    void beginSample() {
        assert(!_inSample && "beginSample() called while a sample is already in progress (missing endSample()?)");
        _current.clear();
        _cursor = 0;
        _inSample = true;
        _callStart = Clock::now();
        _lastSplit = _callStart;
    }

    void section(const std::string& label) {
        assert(_inSample && "section() called outside begin/endSample");
        const auto now = Clock::now();
        const ns elapsed = std::chrono::duration_cast<ns>(now - _lastSplit);
        _lastSplit = now;

        if (_samples.empty()) {
            // First call establishes the section order.
            _labels.push_back(label);
        } else {
            // Subsequent calls must hit the same labels in the same order so
            // that _samples rows stay column-aligned with _labels.
            assert(_cursor < _labels.size() && _labels[_cursor] == label &&
                   "section labels/order must match across calls");
        }
        _current.push_back(elapsed);
        ++_cursor;
    }

    void endSample() {
        assert(_inSample && "endSample() without beginSample()");
        const ns total = std::chrono::duration_cast<ns>(Clock::now() - _callStart);
        _samples.push_back(_current);
        _totals.push_back(total);
        _inSample = false;
    }

    struct Stats {
        std::size_t count = 0;
        ns min{0};
        ns max{0};
        ns sum{0};
        ns mean{0};
        ns median{0};
        ns stddev{0};
    };

    const std::vector<std::string>& labels() const { return _labels; }
    std::size_t numSamples() const { return _samples.size(); }

    // Aggregate one section (by label) across all recorded calls.
    Stats statsFor(const std::string& label) const {
        for (std::size_t s = 0; s < _labels.size(); ++s) {
            if (_labels[s] == label) return statsAtColumn(s);
        }
        return {};
    }

    // Aggregate the per-call totals across all recorded calls.
    Stats totalStats() const { return statsFromValues(_totals); }

    // Human-readable table: one row per section plus a TOTAL row.
    std::string summary(const std::string& title) const {
        std::ostringstream out;
        out << "=== " << title << " (" << numSamples() << " calls) ===\n";
        if (numSamples() == 0) {
            out << "  (no samples recorded)\n";
            return out.str();
        }

        const ns meanTotal = totalStats().mean;

        auto writeRow = [&](const std::string& name, const Stats& st, bool showPct) {
            const auto [meanV, meanU] = formatDuration(st.mean);
            const auto [minV, minU] = formatDuration(st.min);
            const auto [maxV, maxU] = formatDuration(st.max);
            const auto [medV, medU] = formatDuration(st.median);
            const auto [sdV, sdU] = formatDuration(st.stddev);
            out << std::fixed << std::setprecision(3);
            out << "  " << std::left << std::setw(38) << name << std::right
                << "mean " << std::setw(9) << meanV << ' ' << std::setw(3) << meanU
                << " | min " << std::setw(9) << minV << ' ' << std::setw(3) << minU
                << " | max " << std::setw(9) << maxV << ' ' << std::setw(3) << maxU
                << " | med " << std::setw(9) << medV << ' ' << std::setw(3) << medU
                << " | sd " << std::setw(9) << sdV << ' ' << std::setw(3) << sdU;
            if (showPct && meanTotal.count() > 0) {
                const double pct = 100.0 * static_cast<double>(st.mean.count()) /
                                   static_cast<double>(meanTotal.count());
                out << " | " << std::setprecision(1) << pct << "%";
            }
            out << '\n';
        };

        for (std::size_t s = 0; s < _labels.size(); ++s) {
            writeRow(_labels[s], statsAtColumn(s), true);
        }
        writeRow("TOTAL", totalStats(), false);
        return out.str();
    }

    // Long-format rows: call,section,duration_ns — one row per (call, section)
    // plus a TOTAL pseudo-section per call.
    void writeCsv(std::ostream& os) const {
        os << "call,section,duration_ns\n";
        for (std::size_t call = 0; call < _samples.size(); ++call) {
            const auto& sample = _samples[call];
            for (std::size_t s = 0; s < sample.size(); ++s) {
                os << call << ',' << _labels[s] << ',' << sample[s].count() << '\n';
            }
            os << call << ",TOTAL," << _totals[call].count() << '\n';
        }
    }

   private:
    Stats statsAtColumn(std::size_t col) const {
        std::vector<ns> values;
        values.reserve(_samples.size());
        for (const auto& sample : _samples) {
            if (col < sample.size()) values.push_back(sample[col]);
        }
        return statsFromValues(values);
    }

    static Stats statsFromValues(const std::vector<ns>& valuesIn) {
        Stats st;
        if (valuesIn.empty()) return st;

        std::vector<ns> values = valuesIn;
        std::sort(values.begin(), values.end());

        st.count = values.size();
        st.min = values.front();
        st.max = values.back();

        long double sum = 0.0L;
        for (const ns v : values) sum += static_cast<long double>(v.count());
        st.sum = ns{static_cast<ns::rep>(sum)};
        const long double meanD = sum / static_cast<long double>(st.count);
        st.mean = ns{static_cast<ns::rep>(meanD)};

        const std::size_t mid = st.count / 2;
        st.median = (st.count % 2 == 0)
                        ? ns{(values[mid - 1].count() + values[mid].count()) / 2}
                        : values[mid];

        if (st.count > 1) {
            long double acc = 0.0L;
            for (const ns v : values) {
                const long double d = static_cast<long double>(v.count()) - meanD;
                acc += d * d;
            }
            const long double var = acc / static_cast<long double>(st.count - 1);
            st.stddev = ns{static_cast<ns::rep>(std::sqrt(var))};
        }
        return st;
    }
};
