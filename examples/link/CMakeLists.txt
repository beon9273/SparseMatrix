cmake_minimum_required(VERSION 3.16)
project(sparsemat_link_example VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(link_example main.cpp)

# Standalone usage: consume installed package.
# In-tree usage (via add_subdirectory from the main project): reuse existing target.
if(TARGET sparsemat::sparsemat)
	target_link_libraries(link_example PRIVATE sparsemat::sparsemat)
else()
	find_package(sparsemat REQUIRED)
	target_link_libraries(link_example PRIVATE sparsemat::sparsemat)
endif()
