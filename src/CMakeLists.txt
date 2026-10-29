# Everything an executable in this project needs: headers included relative to this
# directory (e.g. "container/CpuAoSContainer.hpp"), the precision switch, the YAML
# parser, and the selected backend's define, toolchain flags and libraries. A build
# directory holds exactly one backend, so one target covers all of it.
add_library(lc_lib INTERFACE)
target_include_directories(lc_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(lc_lib INTERFACE LC_USE_64_BIT=$<BOOL:${USE_64_BIT}>)
target_link_libraries(lc_lib INTERFACE fkYAML::fkYAML)

if(LC_BACKEND STREQUAL "kokkos")
  # Kokkos::kokkos also supplies the -x cuda flags that put clang into CUDA mode.
  target_compile_definitions(lc_lib INTERFACE LC_BACKEND_KOKKOS)
  target_link_libraries(lc_lib INTERFACE Kokkos::kokkos)
elseif(LC_BACKEND STREQUAL "cuda")
  # No Kokkos in this build at all; clang is put into CUDA mode here instead.
  target_compile_definitions(lc_lib INTERFACE LC_BACKEND_CUDA)
  target_compile_options(lc_lib INTERFACE
    -x cuda --cuda-path=${LC_CUDA_PATH} --cuda-gpu-arch=${LC_CUDA_ARCH})
  target_link_libraries(lc_lib INTERFACE CUDA::cudart)
elseif(LC_BACKEND_IS_SYCL)
  # add_sycl_to_target routes a target's sources through acpp; it must be called
  # on each consuming target, after its last target_sources/definitions call.
  # LC_SYCL_FLOW only tags the run; both flows compile identical sources.
  target_compile_definitions(lc_lib INTERFACE LC_BACKEND_SYCL
                             LC_SYCL_FLOW="${LC_SYCL_FLOW}")
endif()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE lc_lib)

if(LC_BACKEND_IS_SYCL)
  add_sycl_to_target(TARGET main SOURCES main.cpp)
endif()
