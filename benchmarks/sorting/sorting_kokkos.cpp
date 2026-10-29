#include <chrono>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "common.hpp"

#include "util/KokkosDefinitions.hpp"

using KeyView = Kokkos::View<uint_type*, Device>;
// LayoutRight keeps each of the num_values arrays contiguous
using ValueView = Kokkos::View<float_type**, Kokkos::LayoutRight, Device>;
using Policy = Kokkos::RangePolicy<Device, Kokkos::IndexType<uint_type>>;

template <typename T>
using HostInput = Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void resetDeviceData(
    const KeyView& keys,
    const ValueView& values,
    const std::vector<uint_type>& initial_keys,
    const std::vector<std::vector<float_type>>& initial_values
) {
    Kokkos::deep_copy(keys, HostInput<uint_type>(initial_keys.data(), initial_keys.size()));
    for (uint_type v = 0; v < initial_values.size(); ++v) {
        Kokkos::deep_copy(Kokkos::subview(values, v, Kokkos::ALL),
                          HostInput<float_type>(initial_values[v].data(), initial_values[v].size()));
    }
    Kokkos::fence();
}

/**
 * Keys must be non-decreasing and every payload must still sit next to the key it
 * started with: values hold the initial position, so initial_keys[value] is that key.
 */
bool checkSorted(const KeyView& keys, const ValueView& values, const std::vector<uint_type>& initial_keys) {
    const auto h_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), keys);
    const auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const uint_type n = keys.extent(0);

    for (uint_type i = 1; i < n; ++i) {
        if (h_keys(i - 1) > h_keys(i)) return false;
    }
    for (uint_type v = 0; v < values.extent(0); ++v) {
        for (uint_type i = 0; i < n; ++i) {
            const auto origin = static_cast<uint_type>(h_values(v, i));
            if (origin >= n || initial_keys[origin] != h_keys(i)) return false;
        }
    }
    return true;
}

/**
 * Kokkos::BinSort benchmarking, one sort per array.
 */
BenchmarkResult benchmarkBinsort(
    const std::vector<uint_type>& keys,
    const std::vector<std::vector<float_type>>& values,
    const Options& opts
) {
    BenchmarkResult result{"binsort", {}};

    KeyView d_keys(Kokkos::view_alloc("keys", Kokkos::WithoutInitializing), opts.n);
    ValueView d_values(Kokkos::view_alloc("values", Kokkos::WithoutInitializing), opts.num_values, opts.n);

    using BinOp = Kokkos::BinOp1D<KeyView>;
    using BinSort = Kokkos::BinSort<KeyView, BinOp, typename KeyView::device_type, uint_type>;
    // BinOp1D adds one bin internally, so this range maps every key onto the bin of the same number.
    const BinOp bin_op(opts.num_bins, 0, opts.num_bins);

    for (uint_type rep = 0; rep < opts.repetitions + opts.warmup; ++rep) {
        resetDeviceData(d_keys, d_values, keys, values);

        const auto started = std::chrono::steady_clock::now();

        BinSort bin_sort(d_keys, bin_op, false);
        bin_sort.create_permute_vector();
        bin_sort.sort(d_keys);
        for (uint_type v = 0; v < opts.num_values; ++v) {
            bin_sort.sort(Kokkos::subview(d_values, v, Kokkos::ALL));
        }
        Kokkos::fence();

        const auto stopped = std::chrono::steady_clock::now();
        if (rep >= opts.warmup) {
            result.record(std::chrono::duration_cast<std::chrono::nanoseconds>(stopped - started).count());
        }

        if (rep == 0 && !checkSorted(d_keys, d_values, keys)) {
            throw std::runtime_error("binsort produced an unsorted result");
        }
    }
    return result;
}

/**
 * Manual counting sort benchmarking
 */
BenchmarkResult benchmarkManualSort(
    const std::vector<uint_type>& keys,
    const std::vector<std::vector<float_type>>& values,
    const Options& opts
) {
    BenchmarkResult result{"manual", {}};

    KeyView d_keys(Kokkos::view_alloc("keys", Kokkos::WithoutInitializing), opts.n);
    KeyView d_keys_tmp(Kokkos::view_alloc("keys_tmp", Kokkos::WithoutInitializing), opts.n);
    ValueView d_values(Kokkos::view_alloc("values", Kokkos::WithoutInitializing), opts.num_values, opts.n);
    ValueView d_values_tmp(Kokkos::view_alloc("values_tmp", Kokkos::WithoutInitializing), opts.num_values, opts.n);
    KeyView perm(Kokkos::view_alloc("perm", Kokkos::WithoutInitializing), opts.n);
    KeyView offsets("bin_offsets", opts.num_bins + 1);
    KeyView cursor("bin_cursor", opts.num_bins);

    const uint_type n = opts.n;
    const uint_type num_bins = opts.num_bins;
    const uint_type num_values = opts.num_values;

    for (uint_type rep = 0; rep < opts.repetitions + opts.warmup; ++rep) {
        resetDeviceData(d_keys, d_values, keys, values);

        const KeyView src_keys = d_keys;
        const KeyView dst_keys = d_keys_tmp;
        const ValueView src_values = d_values;
        const ValueView dst_values = d_values_tmp;

        const auto started = std::chrono::steady_clock::now();

        // Counting bins
        Kokkos::deep_copy(offsets, uint_type(0));
        Kokkos::parallel_for("CountBins", Policy(0, n), KOKKOS_LAMBDA(uint_type i) {
            Kokkos::atomic_fetch_add(&offsets(src_keys(i)), uint_type(1));
        });

        // Exclusive prefix sum. Bin b bounds are [offsets(b), offsets(b+1)).
        Kokkos::parallel_scan("ScanBins", Policy(0, num_bins + 1),
            KOKKOS_LAMBDA(uint_type b, uint_type& running, const bool final) {
                const uint_type count = offsets(b);
                if (final) offsets(b) = running;
                running += count;
            });

        // Scatter values to their new positions, using the offsets as write cursors
        Kokkos::deep_copy(cursor, Kokkos::subview(offsets, Kokkos::make_pair(uint_type(0), num_bins)));
        Kokkos::parallel_for("ScatterPerm", Policy(0, n), KOKKOS_LAMBDA(uint_type old_i) {
            const uint_type new_i = Kokkos::atomic_fetch_add(&cursor(src_keys(old_i)), uint_type(1));
            perm(new_i) = old_i;
        });

        // Reorder keys and values according to the permutation
        Kokkos::parallel_for("Reorder", Policy(0, n), KOKKOS_LAMBDA(uint_type new_i) {
            const uint_type old_i = perm(new_i);
            dst_keys(new_i) = src_keys(old_i);
            for (uint_type v = 0; v < num_values; ++v) {
                dst_values(v, new_i) = src_values(v, old_i);
            }
        });
        Kokkos::fence();

        const auto stopped = std::chrono::steady_clock::now();
        if (rep >= opts.warmup) {
            result.record(std::chrono::duration_cast<std::chrono::nanoseconds>(stopped - started).count());
        }

        std::swap(d_keys, d_keys_tmp);
        std::swap(d_values, d_values_tmp);

        if (rep == 0 && !checkSorted(d_keys, d_values, keys)) {
            throw std::runtime_error("manual sort produced an unsorted result");
        }
    }
    return result;
}

int main(int argc, const char* argv[]) {
    const Options opts = parseArgs(argc, argv);

    std::vector<uint_type> keys;
    std::vector<std::vector<float_type>> values(opts.num_values);
    initializeSortingData(keys, values, opts);

    Kokkos::initialize(argc, const_cast<char**>(argv));
    {
        const BenchmarkResult binsort_result = benchmarkBinsort(keys, values, opts);
        const BenchmarkResult manual_sort_result = benchmarkManualSort(keys, values, opts);

        binsort_result.print_statistics();
        manual_sort_result.print_statistics();

        binsort_result.write_csv(opts);
        manual_sort_result.write_csv(opts);
    }
    Kokkos::finalize();
    return 0;
}
