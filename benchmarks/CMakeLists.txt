# Standalone benchmark executables for research and exploration. They are not
# part of the test suite. lc_lib provides the src/ include path plus the Kokkos
# toolchain and libraries.

if(LC_BACKEND STREQUAL "kokkos")
  add_executable(sorting sorting/sorting_kokkos.cpp)
  target_link_libraries(sorting PRIVATE lc_lib)
endif()
