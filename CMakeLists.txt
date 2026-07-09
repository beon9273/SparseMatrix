cmake_minimum_required(VERSION 3.16)
project(sparsemat VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optional CUDA support ---
# Off by default: only enable when a CUDA-capable toolchain is present and the
# user opts in, so a plain CPU checkout never needs nvcc installed.
option(SPARSEMAT_ENABLE_CUDA "Build CUDA examples (requires nvcc)" OFF)
if(SPARSEMAT_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
  else()
    message(WARNING "SPARSEMAT_ENABLE_CUDA requested but no CUDA compiler was found; skipping CUDA targets.")
    set(SPARSEMAT_ENABLE_CUDA OFF)
  endif()
endif()

enable_testing()

add_executable(test_sparsemat tests/test_sparsemat.cpp)
add_dependencies(test_sparsemat dist)
target_include_directories(test_sparsemat PRIVATE ${CMAKE_SOURCE_DIR}/dist)
target_compile_options(test_sparsemat PRIVATE -Wall -Wextra -Werror)

add_test(NAME sparsemat_tests COMMAND test_sparsemat)

# --- GPU test runner ---
# Every SparseMat operation is marked SPARSEMAT_HD (__host__ __device__).
# test_sparsemat_gpu.cu #includes tests/test_sparsemat.cpp directly and
# launches a CUDA kernel that calls its run_all_tests() — the exact same
# test_*() functions the CPU runner (test_sparsemat above) uses, just called
# from device code instead of from a host main(). One set of tests, two
# runners: this both proves the code compiles for the device (the whole
# binary wouldn't build otherwise) and that it runs correctly there (the
# kernel actually executes the checks and reports pass/fail back).
# Skips (rather than fails) when no physical GPU is available, e.g. in CI.
if(SPARSEMAT_ENABLE_CUDA)
  add_executable(test_sparsemat_gpu tests/test_sparsemat_gpu.cu)
  add_dependencies(test_sparsemat_gpu dist)
  target_include_directories(test_sparsemat_gpu PRIVATE ${CMAKE_SOURCE_DIR}/dist)
  target_compile_options(test_sparsemat_gpu PRIVATE --expt-relaxed-constexpr)
  add_test(NAME sparsemat_tests_gpu COMMAND test_sparsemat_gpu)
  set_tests_properties(sparsemat_tests_gpu PROPERTIES SKIP_RETURN_CODE 99)
endif()

# --- Coverage ---
# Builds a second, instrumented copy of the test binary against the original
# include/ headers (not the dist/ amalgamation) so gcovr can attribute
# coverage to the real source files. Off by default since --coverage disables
# optimization and bloats binaries; opt in with -DSPARSEMAT_ENABLE_COVERAGE=ON.
option(SPARSEMAT_ENABLE_COVERAGE "Build an instrumented test target and a 'coverage' report target (requires gcovr)" OFF)
if(SPARSEMAT_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "SPARSEMAT_ENABLE_COVERAGE requires GCC or Clang.")
  endif()

  find_program(GCOVR_EXECUTABLE gcovr)
  if(NOT GCOVR_EXECUTABLE)
    message(WARNING "gcovr not found (pip install gcovr); the 'coverage' target will be unavailable.")
  endif()

  add_executable(test_sparsemat_coverage tests/test_sparsemat.cpp)
  target_include_directories(test_sparsemat_coverage PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/sparsemat/api
  )
  target_compile_options(test_sparsemat_coverage PRIVATE -Wall -Wextra -Werror -O0 -g --coverage)
  target_link_options(test_sparsemat_coverage PRIVATE --coverage)

  if(GCOVR_EXECUTABLE)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
      COMMAND $<TARGET_FILE:test_sparsemat_coverage>
      COMMAND ${GCOVR_EXECUTABLE}
              --root ${CMAKE_SOURCE_DIR}
              --filter "${CMAKE_SOURCE_DIR}/include/.*"
              --object-directory ${CMAKE_BINARY_DIR}
              --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
              --print-summary
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS test_sparsemat_coverage
      COMMENT "Running instrumented tests and generating coverage report at build/coverage/index.html"
      VERBATIM
    )
  endif()
endif()

# --- Dist (single-header amalgamation) ---

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(dist
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
  COMMAND chmod -f u+w ${CMAKE_SOURCE_DIR}/dist/sparsemat.h || true
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/amalgamate.py
          ${CMAKE_SOURCE_DIR}/include/sparsemat/api/sparsemat.h
          ${CMAKE_SOURCE_DIR}/dist/sparsemat.h
          ${CMAKE_SOURCE_DIR}/include
  COMMAND chmod 444 ${CMAKE_SOURCE_DIR}/dist/sparsemat.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Building dist/sparsemat.h (read-only, generated file)"
  VERBATIM
)

# --- Install / package ---

# Interface library — header-only, no compilation needed by consumers
add_library(sparsemat INTERFACE)
add_library(sparsemat::sparsemat ALIAS sparsemat)

target_include_directories(sparsemat INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(sparsemat INTERFACE cxx_std_20)

option(SPARSEMAT_BUILD_EXAMPLES "Build examples" ON)
if(SPARSEMAT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# --- Compile-fail tests for SparseMat::validate_indicies() ---
# Duplicate/out-of-bounds sparsity indices are rejected via static_assert, so
# the only way to test them is to confirm the offending translation unit
# fails to compile. A control case (unique, in-bounds indices) confirms these
# tests fail for the *expected* reason rather than a broken include path.
function(sparsemat_add_compile_test name expect_success)
  try_compile(SPARSEMAT_CT_${name}
    ${CMAKE_BINARY_DIR}/compile_tests/${name}
    ${CMAKE_SOURCE_DIR}/tests/compile_fail/${name}.cpp
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${CMAKE_SOURCE_DIR}/include
  )
  if(SPARSEMAT_CT_${name} STREQUAL expect_success)
    add_test(NAME compile_${name} COMMAND ${CMAKE_COMMAND} -E true)
  else()
    add_test(NAME compile_${name} COMMAND ${CMAKE_COMMAND} -E false)
  endif()
endfunction()

sparsemat_add_compile_test(valid_indices_control TRUE)
sparsemat_add_compile_test(duplicate_index FALSE)
sparsemat_add_compile_test(out_of_bounds_index FALSE)
sparsemat_add_compile_test(negative_index FALSE)
sparsemat_add_compile_test(negative_dimensions FALSE)
sparsemat_add_compile_test(dot_mismatched_length FALSE)

include(GNUInstallDirs)

install(TARGETS sparsemat
  EXPORT sparsemat_targets
)

install(DIRECTORY include/sparsemat/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sparsemat
  FILES_MATCHING PATTERN "*.h"
)

install(EXPORT sparsemat_targets
  FILE sparsemat-targets.cmake
  NAMESPACE sparsemat::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparsemat
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sparsemat-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sparsemat-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparsemat
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/sparsemat-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/sparsemat-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/sparsemat-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparsemat
)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYGEN_PROJECT_NAME "sparsemat")
  set(DOXYGEN_PROJECT_BRIEF "Compile-time sparse matrix library")
  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
  set(DOXYGEN_TEMPLATE_RELATIONS YES)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_LATEX NO)
  set(DOXYGEN_QUIET YES)

  doxygen_add_docs(docs
    ${CMAKE_SOURCE_DIR}/include/sparsemat/api/sparsemat.h
    ${CMAKE_SOURCE_DIR}/include/sparsemat/operations
    COMMENT "Generating API documentation"
  )
endif()
