#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "util/TypeDefinitions.hpp"

inline void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " -n <count> [options]\n"
              << "  -n <count>    number of elements (required)\n"
              << "  -b <count>    number of bins (default 4096)\n"
              << "  -v <count>    number of value arrays (default 1)\n"
              << "  -r <count>    timed repetitions (default 100)\n"
              << "  -w <count>    warmup repetitions (default 5)\n"
              << "  -o <dir>      output directory for CSV results\n"
              << "  -h            show this help" << std::endl;
}

struct Options {
    uint_type n;
    uint_type num_bins = 4096;
    uint_type num_values = 1;
    uint_type repetitions = 100;
    uint_type warmup = 5;
    std::string output_path;
};

inline Options parseArgs(int argc, const char* argv[]) {
    Options opts;
    bool has_n = false;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h") {
            printUsage(argv[0]);
            exit(0);
        }
        if (flag.size() != 2 || flag[0] != '-') {
            std::cerr << "Error: unexpected argument: " << flag << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
        if (i + 1 >= argc) {
            std::cerr << "Error: missing value for option: " << flag << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
        const std::string value = argv[++i];

        if (flag == "-o") {
            opts.output_path = value;
            continue;
        }

        uint_type parsed;
        try {
            parsed = static_cast<uint_type>(std::stoull(value));
        } catch (const std::exception&) {
            std::cerr << "Error: invalid number for option " << flag << ": " << value << std::endl;
            printUsage(argv[0]);
            exit(1);
        }

        switch (flag[1]) {
            case 'n': opts.n = parsed; has_n = true; break;
            case 'b': opts.num_bins = parsed; break;
            case 'v': opts.num_values = parsed; break;
            case 'r': opts.repetitions = parsed; break;
            case 'w': opts.warmup = parsed; break;
            default:
                std::cerr << "Error: unknown option: " << flag << std::endl;
                printUsage(argv[0]);
                exit(1);
        }
    }

    if (!has_n) {
        std::cerr << "Error: missing required option -n" << std::endl;
        printUsage(argv[0]);
        exit(1);
    }
    return opts;
}


inline void initializeSortingData(
    std::vector<uint_type>& keys,
    std::vector<std::vector<float_type>>& values,
    const Options& opts
) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint_type> bin(0, opts.num_bins - 1);

    keys.resize(opts.n);
    values[0].resize(opts.n);
    for (uint_type i = 0; i < opts.n; ++i) {
        keys[i] = bin(rng);
        values[0][i] = static_cast<float_type>(i);
    }
    for (uint_type i = 1; i < opts.num_values; ++i) {
        values[i] = values[0];
    }
}

inline std::string makeRunTitle(const Options& opts) {
    return "n" + std::to_string(opts.n) + "_b" + std::to_string(opts.num_bins) + "_v" +
           std::to_string(opts.num_values);
}

struct BenchmarkResult {
    std::string name;
    std::vector<long> nanoseconds;

    void record(long ns) {
        nanoseconds.push_back(ns);
    }

    void print_statistics() const {
        if (nanoseconds.empty()) {
            std::cout << "--- Benchmark: " << name << " --- (no samples)" << std::endl;
            return;
        }

        long min_nanoseconds = *std::min_element(nanoseconds.begin(), nanoseconds.end());
        long max_nanoseconds = *std::max_element(nanoseconds.begin(), nanoseconds.end());
        double sum = 0;
        for (const auto& t : nanoseconds) {
            sum += static_cast<double>(t);
        }
        double mean_nanoseconds = sum / static_cast<double>(nanoseconds.size());
        double sum_sq = 0;
        for (const auto& t : nanoseconds) {
            double diff = static_cast<double>(t) - mean_nanoseconds;
            sum_sq += diff * diff;
        }
        double stddev_nanoseconds = std::sqrt(sum_sq / static_cast<double>(nanoseconds.size()));

        std::cout << "--- Benchmark: " << name << " ---" << std::endl;
        std::cout << "Mean time (ns): " << mean_nanoseconds << std::endl;
        std::cout << "Stddev time (ns): " << stddev_nanoseconds << std::endl;
        std::cout << "Min time (ns): " << min_nanoseconds << std::endl;
        std::cout << "Max time (ns): " << max_nanoseconds << std::endl;
    }

    void write_csv(const Options& opts) const {
        if (opts.output_path.empty()) return;

        const std::string filename = opts.output_path + "/" + makeRunTitle(opts) + "_" + name + ".csv";
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Error: could not open file for writing: " << filename << std::endl;
            return;
        }
        ofs << "run,nanoseconds\n";
        for (size_t i = 0; i < nanoseconds.size(); ++i) {
            ofs << i << "," << nanoseconds[i] << "\n";
        }
        std::cout << "Benchmark results written to: " << filename << std::endl;
    }
};
