cmake_minimum_required(VERSION 3.22)
project(LinkedCellsGPU CXX)

# THIS IS THE MAGIC LINE FOR INTELLISENSE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Require C++20 (Kokkos requires modern C++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Switches from 32-bit to 64-bit types
option(USE_64_BIT "Use 64-bit double precision and 64-bit particle ids (OFF builds with 32-bit float and uint32_t ids)" OFF)

# Exactly one backend per build directory, each with its own toolchain: Kokkos
# (clang in CUDA mode via Kokkos' own flags), native CUDA (clang in CUDA mode,
# no Kokkos anywhere) and SYCL (acpp). 
# The sources are shared; Simulation.hpp selects containers and traversals on LC_BACKEND_<NAME>.
# The two SYCL builds share every source and differ only in the ACPP target they
# compile for, so they are separate backends purely to keep their results apart.
set(LC_BACKEND "kokkos" CACHE STRING "Backend to build: kokkos | cuda | sycl-cuda | sycl-generic")
set_property(CACHE LC_BACKEND PROPERTY STRINGS kokkos cuda sycl-cuda sycl-generic)
if(NOT LC_BACKEND MATCHES "^(kokkos|cuda|sycl-cuda|sycl-generic)$")
  message(FATAL_ERROR "Invalid LC_BACKEND '${LC_BACKEND}' "
                      "(expected kokkos, cuda, sycl-cuda or sycl-generic)")
endif()
if(LC_BACKEND MATCHES "^sycl-")
  set(LC_BACKEND_IS_SYCL TRUE)
else()
  set(LC_BACKEND_IS_SYCL FALSE)
endif()

# GPU architecture for the native CUDA build and the native CUDA benchmarks.
set(LC_CUDA_ARCH "sm_86" CACHE STRING "GPU architecture for the native CUDA targets")

include(FetchContent)

# Header-only YAML parser for config-file input (src/io/ConfigFileLoader.hpp).
FetchContent_Declare(
  fkYAML
  URL https://github.com/fktn-k/fkYAML/archive/refs/tags/v0.4.3.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fkYAML)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  get_filename_component(LC_CUDA_PATH "${CUDAToolkit_BIN_DIR}" DIRECTORY)
endif()

if(LC_BACKEND STREQUAL "kokkos")
  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/5.1.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

if(LC_BACKEND STREQUAL "cuda" AND NOT (CUDAToolkit_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  message(FATAL_ERROR "LC_BACKEND=cuda needs the CUDA toolkit and a clang host compiler "
                      "(clang compiles the .cu-style sources in CUDA mode); "
                      "found CUDAToolkit=${CUDAToolkit_FOUND}, compiler=${CMAKE_CXX_COMPILER_ID}")
endif()

if(LC_BACKEND_IS_SYCL)
  # The backend choice fixes the ACPP target, so the two cannot disagree.
  # LC_SYCL_FLOW is the same choice as a string, for the run tag.
  if(LC_BACKEND STREQUAL "sycl-cuda")
    set(LC_SYCL_FLOW "cuda")
    set(ACPP_TARGETS "cuda:${LC_CUDA_ARCH}" CACHE STRING "AdaptiveCpp target" FORCE)
  else()
    set(LC_SYCL_FLOW "generic")
    set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp target" FORCE)
  endif()

  find_package(AdaptiveCpp CONFIG REQUIRED)
  message(STATUS "SYCL backend; ACPP_TARGETS=${ACPP_TARGETS}")
endif()

enable_testing()

add_subdirectory(src)
# Guards its own targets per backend, so it is added for every build type.
add_subdirectory(benchmarks)
add_subdirectory(test)
