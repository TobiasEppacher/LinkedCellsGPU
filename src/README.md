# CMake Generation

With CUDA:

```sh
cmake ../src -DKokkos_ENABLE_SERIAL=On -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_AMPERE86=On -DKokkos_ENABLE_CUDA_LAMBDA=On
```

Without CUDA:

```sh
cmake ../src -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DKokkos_ENABLE_SERIAL=On -DKokkos_ENABLE_OPENMP=On 
```