#Demo using the full include direcotry
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(demo PRIVATE -Wall -Wextra -Werror)

# Example using the dist include file.
add_executable(kalman kalman.cpp)
add_dependencies(kalman dist)
target_include_directories(kalman PRIVATE ${CMAKE_SOURCE_DIR}/dist ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kalman PRIVATE -Wall -Wextra -Werror)

if(SPARSEMAT_ENABLE_CUDA)
  add_executable(kalman_gpu kalman_gpu.cu)
  add_dependencies(kalman_gpu dist)
  target_include_directories(kalman_gpu PRIVATE ${CMAKE_SOURCE_DIR}/dist ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(kalman_gpu PRIVATE --expt-relaxed-constexpr)
endif()

find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
  add_executable(benchmark bench.cpp)
  add_dependencies(benchmark dist)
  target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/dist ${EIGEN3_INCLUDE_DIRS})
  target_compile_options(benchmark PRIVATE -O3 -march=native)

  # Runs `benchmark` and splices its output into the README's benchmark
  # table (between the BENCHMARK_TABLE markers). Not wired into `all` or
  # `dist`, unlike the amalgamation: benchmark numbers are hardware-dependent
  # and meant to be refreshed by hand, not enforced as up to date in CI.
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_custom_target(update-benchmarks
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/update_benchmark_table.py
              --binary $<TARGET_FILE:benchmark>
              --readme ${CMAKE_SOURCE_DIR}/README.md
      DEPENDS benchmark
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Regenerating the benchmark table in README.md"
      VERBATIM
    )
  endif()
endif()

# Build the package-consumer link example as part of the examples suite.
add_subdirectory(link)
