# Each test is a plain executable that returns non-zero on failure and is
# registered with CTest. Run the whole suite with: ctest --output-on-failure
# (from the build directory). One test per backend, each validating against the
# shared O(N^2) reference in ParticleFixture.hpp; agreeing with that reference is
# what ties the backends to each other.

if(LC_BACKEND STREQUAL "kokkos")
  add_executable(kokkos_force_test kokkos_force_test.cpp)
  target_include_directories(kokkos_force_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(kokkos_force_test PRIVATE lc_lib)
  add_test(NAME kokkos_force_test COMMAND kokkos_force_test)
elseif(LC_BACKEND STREQUAL "cuda")
  add_executable(cuda_force_test cuda_force_test.cpp)
  target_include_directories(cuda_force_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cuda_force_test PRIVATE lc_lib)
  add_test(NAME cuda_force_test COMMAND cuda_force_test)
elseif(LC_BACKEND_IS_SYCL)
  add_executable(sycl_force_test sycl_force_test.cpp)
  target_include_directories(sycl_force_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(sycl_force_test PRIVATE lc_lib)
  add_sycl_to_target(TARGET sycl_force_test SOURCES sycl_force_test.cpp)
  add_test(NAME sycl_force_test COMMAND sycl_force_test)
endif()
