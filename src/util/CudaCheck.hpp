#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Throw on a failed CUDA call so errors surface immediately with context.
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t _cuda_status = (call);                                              \
        if (_cuda_status != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") +                            \
                                     cudaGetErrorString(_cuda_status) + " at " __FILE__ ":" + \
                                     std::to_string(__LINE__));                               \
        }                                                                                     \
    } while (0)
