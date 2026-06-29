cmake_minimum_required(VERSION 3.16)
project(sparsemat VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_sparsemat tests/test_sparsemat.cpp)
add_dependencies(test_sparsemat dist)
target_include_directories(test_sparsemat PRIVATE ${CMAKE_SOURCE_DIR}/dist)
target_compile_options(test_sparsemat PRIVATE -Wall -Wextra -Werror)

add_test(NAME sparsemat_tests COMMAND test_sparsemat)

# --- Dist (single-header amalgamation) ---

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(dist
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/amalgamate.py
          ${CMAKE_SOURCE_DIR}/include/sparsemat/api/sparsemat.h
          ${CMAKE_SOURCE_DIR}/dist/sparsemat.h
          ${CMAKE_SOURCE_DIR}/include
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Building dist/sparsemat.h"
  VERBATIM
)

option(SPARSEMAT_BUILD_EXAMPLES "Build examples" ON)
if(SPARSEMAT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# --- Install / package ---

# Interface library — header-only, no compilation needed by consumers
add_library(sparsemat INTERFACE)
add_library(sparsemat::sparsemat ALIAS sparsemat)

target_include_directories(sparsemat INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(sparsemat INTERFACE cxx_std_20)

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
