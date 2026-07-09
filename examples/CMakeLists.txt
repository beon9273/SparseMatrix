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

# --- GPU throughput benchmark ---
# Compares sparsemat vs Eigen dense, both running one instance per CUDA
# thread over a batch of small matrices, against the same sparsemat
# workload run single-threaded on the CPU. Needs both a CUDA toolchain
# (for the device kernels) and Eigen (for the dense-GPU comparison), so it
# only exists when both of the above blocks succeeded.
if(SPARSEMAT_ENABLE_CUDA AND EIGEN3_FOUND)
  add_executable(benchmark_gpu bench_gpu.cu)
  add_dependencies(benchmark_gpu dist)
  target_include_directories(benchmark_gpu PRIVATE ${CMAKE_SOURCE_DIR}/dist ${EIGEN3_INCLUDE_DIRS})
  target_compile_options(benchmark_gpu PRIVATE --expt-relaxed-constexpr)

  # Runs `benchmark_gpu` and splices its output into the README's GPU
  # benchmark table (between the BENCHMARK_TABLE_GPU markers), via the same
  # script used for the CPU table. On a machine with no physical GPU,
  # benchmark_gpu reports "no device" and prints no table; the script
  # detects that and leaves the README untouched instead of failing, so this
  # target is safe to run on any machine that can build it — it just won't
  # produce results until run on one with an actual GPU.
  if(Python3_FOUND)
    add_custom_target(update-gpu-benchmarks
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/update_benchmark_table.py
              --binary $<TARGET_FILE:benchmark_gpu>
              --readme ${CMAKE_SOURCE_DIR}/README.md
              --start-marker "<!-- BENCHMARK_TABLE_GPU:START -->"
              --end-marker "<!-- BENCHMARK_TABLE_GPU:END -->"
      DEPENDS benchmark_gpu
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Regenerating the GPU benchmark table in README.md (no-op without a GPU)"
      VERBATIM
    )
  endif()
endif()

# Build the package-consumer link example as part of the examples suite.
add_subdirectory(link)
