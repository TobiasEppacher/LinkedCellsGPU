#!/bin/bash
# Configures and builds every backend into its own build-<backend>/ directory.
# From the project root:
#                bash scripts/build/build.sh [32|64] [backend ...]
#
#   32|64        bitwidth (USE_64_BIT), default 32
#   backend      any of kokkos cuda sycl-cuda sycl-generic, default all four
#
# Options (environment variables):
#   CXX=clang++  host compiler (e.g. CXX=clang++-18)
#   ARCH=86      GPU compute capability (Kokkos_ARCH_AMPERE86 / sm_86)
#   BUILD_TYPE=Release  CMake build type (e.g. BUILD_TYPE=Debug)

BITS=${1:-32}
shift
BACKENDS=${@:-kokkos cuda sycl-cuda sycl-generic}
ARCH=${ARCH:-86}
[ "$BITS" = 64 ] && USE_64_BIT=ON || USE_64_BIT=OFF

KOKKOS_FLAGS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE${ARCH}=ON"

for backend in $BACKENDS; do
    extra=""
    [ "$backend" = kokkos ] && extra=$KOKKOS_FLAGS
    cmake -S . -B "build-$backend" -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
        -DCMAKE_CXX_COMPILER="${CXX:-clang++}" -DLC_BACKEND="$backend" \
        -DUSE_64_BIT=$USE_64_BIT -DLC_CUDA_ARCH="sm_$ARCH" $extra \
    && cmake --build "build-$backend" -j
done
