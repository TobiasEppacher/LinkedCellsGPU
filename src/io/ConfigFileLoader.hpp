#pragma once

#include <fkYAML/node.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../util/TypeDefinitions.hpp"
#include "AlgorithmType.hpp"
#include "FunctorType.hpp"
#include "InitializerParams.hpp"
#include "SimulationParams.hpp"

/**
 * Thrown for every config-file problem: I/O, YAML syntax, missing or unknown keys and
 * values that do not fit their target type. what() reads "<file>: '<key path>': <reason>".
 */
class ConfigError : public std::runtime_error {
   public:
    explicit ConfigError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * Parses a YAML config file into SimulationParams. Every key is mandatory except
 * 'output_dir' (absent means no output), and unknown keys are rejected. Only the
 * structure and the value types are checked; whether the values make sense as a
 * simulation setup is not. Expected schema:
 *
 *   delta_t: 0.01
 *   end_t: 1.0
 *   domain_size: [10.0, 10.0, 10.0]     # domain is [0, 0, 0] to domain_size
 *   interaction_radius: 1.0
 *   epsilon: 1.0                        # Lennard-Jones well depth
 *   sigma: 0.4                          # Lennard-Jones zero-crossing distance
 *   algorithm: kokkos-c08-soa           # cpu | kokkos-soa | kokkos-soa-newton3 | kokkos-aos |
 *                                       #   kokkos-c01-soa | kokkos-c01-aos |
 *                                       #   kokkos-c01-scratch-soa | kokkos-c01-scratch-aos |
 *                                       #   kokkos-c08-soa | kokkos-c08-scratch-soa |
 *                                       #   cuda-soa | cuda-aos
 *   functor: lj                         # lj | high | low | full
 *   only_forces: false                  # true skips integration and repeats the force calculation
 *   output_vtk_files: true              # per-timestep VTK snapshots; needs output_dir
 *   output_dir: "./output"              # optional
 *   initializers:                       # applied in order
 *     - type: random
 *       num_particles: 1000
 *       box_min: [0.0, 0.0, 0.0]
 *       box_max: [10.0, 10.0, 10.0]
 *     - type: cuboid
 *       box_min: [0.0, 0.0, 0.0]
 *       box_max: [5.0, 5.0, 5.0]
 *       grid_counts: [10, 10, 10]
 */
class ConfigFileLoader {
   public:
    static SimulationParams load(const std::filesystem::path& path) {
        return ConfigFileLoader(path).parse();
    }

   private:
    using InitializerParams = std::variant<RandomInitializerParams, CuboidInitializerParams>;

    std::filesystem::path file_path;

    explicit ConfigFileLoader(std::filesystem::path path) : file_path(std::move(path)) {}

    SimulationParams parse() const {
        const fkyaml::node root = deserialize();
        checkKeys(root, "",
                  {"delta_t", "end_t", "domain_size", "interaction_radius", "epsilon", "sigma",
                   "algorithm", "functor", "only_forces", "output_vtk_files", "initializers"},
                  {"output_dir"});

        SimulationParams params;
        params.input_file_path = file_path;
        params.output_dir = root.contains("output_dir") ? requireString(root, "output_dir", "") : "";
        params.delta_t = requireFloat(root, "delta_t", "");
        params.end_t = requireFloat(root, "end_t", "");
        params.domain_size = requireVec3(root, "domain_size", "");
        params.interaction_radius = requireFloat(root, "interaction_radius", "");
        params.epsilon = requireFloat(root, "epsilon", "");
        params.sigma = requireFloat(root, "sigma", "");
        params.algorithm_type = requireAlgorithmType(root, "algorithm", "");
        params.functor_type = requireFunctorType(root, "functor", "");
        params.only_forces = requireBool(root, "only_forces", "");
        params.output_vtk_files = requireBool(root, "output_vtk_files", "");
        params.initializers = requireInitializers(root);
        return params;
    }

    fkyaml::node deserialize() const {
        std::ifstream file(file_path);
        if (!file) {
            fail("", "could not open config file");
        }
        try {
            return fkyaml::node::deserialize(file);
        } catch (const std::exception& e) {
            fail("", std::string("YAML parse error: ") + e.what());
        }
    }

    std::vector<InitializerParams> requireInitializers(const fkyaml::node& root) const {
        const fkyaml::node& node = child(root, "initializers", "");
        if (!node.is_sequence()) {
            fail("initializers", "must be a sequence of initializer entries");
        }

        std::vector<InitializerParams> initializers;
        initializers.reserve(node.as_seq().size());
        for (size_t i = 0; i < node.as_seq().size(); ++i) {
            initializers.push_back(
                requireInitializer(node.as_seq()[i], "initializers[" + std::to_string(i) + "]"));
        }
        return initializers;
    }

    InitializerParams requireInitializer(const fkyaml::node& node, const std::string& path) const {
        if (!node.is_mapping()) {
            fail(path, "must be a mapping with a 'type' key");
        }
        const std::string type = requireString(node, "type", path);

        if (type == "random") {
            checkKeys(node, path, {"type", "num_particles", "box_min", "box_max"}, {});
            RandomInitializerParams params;
            params.num_particles = requireUint(node, "num_particles", path);
            params.box_min = requireVec3(node, "box_min", path);
            params.box_max = requireVec3(node, "box_max", path);
            return params;
        }
        if (type == "cuboid") {
            checkKeys(node, path, {"type", "box_min", "box_max", "grid_counts"}, {});
            CuboidInitializerParams params;
            params.box_min = requireVec3(node, "box_min", path);
            params.box_max = requireVec3(node, "box_max", path);
            params.grid_counts = requireUintVec3(node, "grid_counts", path);
            return params;
        }
        fail(keyPath(path, "type"), "unknown initializer type '" + type + "' (expected 'random' or 'cuboid')");
    }

    AlgorithmType requireAlgorithmType(const fkyaml::node& parent, const std::string& key,
                                       const std::string& parent_path) const {
        const std::string value = requireString(parent, key, parent_path);
        try {
            return getAlgorithmTypeFromKey(value);
        } catch (const std::invalid_argument&) {
            fail(keyPath(parent_path, key),
                 "unknown algorithm '" + value + "' (expected one of " + keyList(getAlgorithmKeyList()) +
                     ")");
        }
    }

    FunctorType requireFunctorType(const fkyaml::node& parent, const std::string& key,
                                   const std::string& parent_path) const {
        const std::string value = requireString(parent, key, parent_path);
        try {
            return getFunctorTypeFromKey(value);
        } catch (const std::invalid_argument&) {
            fail(keyPath(parent_path, key),
                 "unknown functor '" + value + "' (expected one of " + keyList(getFunctorKeyList()) + ")");
        }
    }

    // --- typed key access; `parent_path` is the key path of the enclosing mapping ---

    const fkyaml::node& child(const fkyaml::node& parent, const std::string& key,
                              const std::string& parent_path) const {
        if (!parent.is_mapping()) {
            fail(parent_path, "must be a mapping");
        }
        if (!parent.contains(key)) {
            fail(parent_path, "missing mandatory key '" + key + "'");
        }
        return parent[key];
    }

    std::string requireString(const fkyaml::node& parent, const std::string& key,
                              const std::string& parent_path) const {
        const fkyaml::node& node = child(parent, key, parent_path);
        if (!node.is_string()) {
            fail(keyPath(parent_path, key), "must be a string");
        }
        return node.as_str();
    }

    bool requireBool(const fkyaml::node& parent, const std::string& key,
                     const std::string& parent_path) const {
        const fkyaml::node& node = child(parent, key, parent_path);
        if (!node.is_boolean()) {
            fail(keyPath(parent_path, key), "must be a boolean");
        }
        return node.as_bool();
    }

    float_type requireFloat(const fkyaml::node& parent, const std::string& key,
                            const std::string& parent_path) const {
        return toFloat(child(parent, key, parent_path), keyPath(parent_path, key));
    }

    uint_type requireUint(const fkyaml::node& parent, const std::string& key,
                          const std::string& parent_path) const {
        return toUint(child(parent, key, parent_path), keyPath(parent_path, key));
    }

    Vec3 requireVec3(const fkyaml::node& parent, const std::string& key, const std::string& parent_path) const {
        const std::string path = keyPath(parent_path, key);
        const auto& entries = requireTriple(parent, key, parent_path, "numbers");
        return {toFloat(entries[0], elementPath(path, 0)), toFloat(entries[1], elementPath(path, 1)),
                toFloat(entries[2], elementPath(path, 2))};
    }

    Vec3u requireUintVec3(const fkyaml::node& parent, const std::string& key,
                          const std::string& parent_path) const {
        const std::string path = keyPath(parent_path, key);
        const auto& entries = requireTriple(parent, key, parent_path, "unsigned integers");
        return {toUint(entries[0], elementPath(path, 0)), toUint(entries[1], elementPath(path, 1)),
                toUint(entries[2], elementPath(path, 2))};
    }

    const fkyaml::node::sequence_type& requireTriple(const fkyaml::node& parent, const std::string& key,
                                                     const std::string& parent_path,
                                                     const char* element_kind) const {
        const fkyaml::node& node = child(parent, key, parent_path);
        if (!node.is_sequence() || node.as_seq().size() != 3) {
            fail(keyPath(parent_path, key), std::string("must be a sequence of 3 ") + element_kind + " [x, y, z]");
        }
        return node.as_seq();
    }

    // --- node value conversion; `path` is the key path of the node itself ---

    float_type toFloat(const fkyaml::node& node, const std::string& path) const {
        if (!node.is_integer() && !node.is_float_number()) {
            fail(path, "must be a number");
        }
        const double value = node.is_integer() ? static_cast<double>(node.as_int()) : node.as_float();
        if (!std::isfinite(value)) {
            fail(path, "must be a finite number");
        }
        // No-op while float_type is double; catches the silent inf/zero results in 32-bit builds.
        if (std::abs(value) > static_cast<double>(std::numeric_limits<float_type>::max())) {
            fail(path, "exceeds the largest representable value");
        }
        if (value != 0 && std::abs(value) < static_cast<double>(std::numeric_limits<float_type>::denorm_min())) {
            fail(path, "underflows to zero");
        }
        return static_cast<float_type>(value);
    }

    uint_type toUint(const fkyaml::node& node, const std::string& path) const {
        if (!node.is_integer()) {
            fail(path, "must be an integer");
        }
        const int64_t value = node.as_int();
        if (value < 0) {
            fail(path, "must not be negative");
        }
        if (static_cast<uint64_t>(value) > std::numeric_limits<uint_type>::max()) {
            fail(path, "exceeds the largest representable value");
        }
        return static_cast<uint_type>(value);
    }

    // --- validation ---

    void checkKeys(const fkyaml::node& map, const std::string& path,
                   std::initializer_list<const char*> mandatory,
                   std::initializer_list<const char*> optional) const {
        if (!map.is_mapping()) {
            fail(path, "must be a mapping");
        }
        for (const char* key : mandatory) {
            if (!map.contains(key)) {
                fail(path, "missing mandatory key '" + std::string(key) + "'");
            }
        }
        for (const auto& entry : map.as_map()) {
            if (!entry.first.is_string()) {
                fail(path, "mapping keys must be strings");
            }
            const std::string& key = entry.first.as_str();
            if (!listed(mandatory, key) && !listed(optional, key)) {
                fail(path, "unknown key '" + key + "'");
            }
        }
    }

    // --- diagnostics ---

    [[noreturn]] void fail(const std::string& path, const std::string& message) const {
        std::string location = file_path.string();
        if (!path.empty()) {
            location += ": '" + path + "'";
        }
        throw ConfigError(location + ": " + message);
    }

    static std::string keyPath(const std::string& parent_path, const std::string& key) {
        return parent_path.empty() ? key : parent_path + "." + key;
    }

    static std::string elementPath(const std::string& path, size_t index) {
        return path + "[" + std::to_string(index) + "]";
    }

    static bool listed(std::initializer_list<const char*> keys, const std::string& key) {
        for (const char* candidate : keys) {
            if (key == candidate) {
                return true;
            }
        }
        return false;
    }

    static std::string keyList(const std::vector<std::string>& keys) {
        std::string list;
        for (const std::string& key : keys) {
            list += (list.empty() ? "" : ", ") + key;
        }
        return list;
    }
};
