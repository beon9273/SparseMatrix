cmake_minimum_required(VERSION 3.16)
project(sparsemat VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_sparsemat tests/test_sparsemat.cpp)
add_dependencies(test_sparsemat dist)
target_include_directories(test_sparsemat PRIVATE ${CMAKE_BINARY_DIR}/dist)
target_compile_options(test_sparsemat PRIVATE -Wall -Wextra -Werror)

add_test(NAME sparsemat_tests COMMAND test_sparsemat)

# --- Dist (single-header amalgamation) ---

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(dist
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/dist
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/amalgamate.py
          ${CMAKE_SOURCE_DIR}/api/sparsemat.h
          ${CMAKE_BINARY_DIR}/dist/sparsemat_dist.h
          ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Building single-header dist/sparsemat.h"
  VERBATIM
)

option(SPARSEMAT_BUILD_EXAMPLES "Build examples" ON)
if(SPARSEMAT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# --- Install / package ---

# Interface library — header-only, no compilation needed by consumers
add_library(sparsemat_lib INTERFACE)
add_library(sparsemat::sparsemat ALIAS sparsemat_lib)

target_include_directories(sparsemat_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/api>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(sparsemat_lib INTERFACE cxx_std_20)

include(GNUInstallDirs)

install(TARGETS sparsemat_lib
  EXPORT sparsemat_targets
)

install(FILES api/sparsemat.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY operations/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operations
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY concepts/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/concepts
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
    ${CMAKE_SOURCE_DIR}/api/sparsemat.h
    ${CMAKE_SOURCE_DIR}/operations
    COMMENT "Generating API documentation"
  )
endif()
