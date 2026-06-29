#Demo using the full include direcotry
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(demo PRIVATE -Wall -Wextra -Werror)

# Example using the dist include file.
add_executable(kalman kalman.cpp)
add_dependencies(kalman dist)
target_include_directories(kalman PRIVATE ${CMAKE_BINARY_DIR}/dist ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kalman PRIVATE -Wall -Wextra -Werror)

find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
  add_executable(benchmark bench.cpp)
  add_dependencies(benchmark dist)
  target_include_directories(benchmark PRIVATE ${CMAKE_BINARY_DIR}/dist ${EIGEN3_INCLUDE_DIRS})
  target_compile_options(benchmark PRIVATE -O3 -march=native)
endif()
