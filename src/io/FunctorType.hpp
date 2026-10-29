#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

enum class FunctorType {
    LENNARD_JONES,
    HIGH_COMPUTE,
    LOW_COMPUTE,
    FULL_PARTICLE
};

inline const std::map<std::string, FunctorType> FUNCTOR_TYPE_KEY_MAP = {
    {"lj", FunctorType::LENNARD_JONES},
    {"high", FunctorType::HIGH_COMPUTE},
    {"low", FunctorType::LOW_COMPUTE},
    {"full", FunctorType::FULL_PARTICLE}
};

inline FunctorType getFunctorTypeFromKey(const std::string& key) {
    auto it = FUNCTOR_TYPE_KEY_MAP.find(key);
    if (it != FUNCTOR_TYPE_KEY_MAP.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Unknown functor key: " + key);
    }
}

inline std::string getKeyFromFunctorType(FunctorType type) {
    for (const auto& pair : FUNCTOR_TYPE_KEY_MAP) {
        if (pair.second == type) {
            return pair.first;
        }
    }
    throw std::invalid_argument("Unknown functor type");
}

inline std::vector<std::string> getFunctorKeyList() {
    std::vector<std::string> keys;
    for (const auto& pair : FUNCTOR_TYPE_KEY_MAP) {
        keys.push_back(pair.first);
    }
    return keys;
}
