#pragma once

/// @file
/// Version macros for the sparsemat library. Kept in sync with the
/// `project(sparsemat VERSION ...)` declaration in CMakeLists.txt — an
/// installed header-only library has no other way to tell a consumer which
/// version it is.
///
/// These are macros rather than `constexpr` constants or an enum (which is what
/// clang-tidy's macro-to-enum and macro-usage checks would otherwise ask for)
/// because their whole purpose is to be usable from the preprocessor: a
/// consumer guarding a feature on the library version has to write
/// `#if SPARSEMAT_VERSION >= SPARSEMAT_VERSION_CHECK(0, 2, 0)`, which no
/// language-level constant can satisfy.

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define SPARSEMAT_VERSION_MAJOR 0
#define SPARSEMAT_VERSION_MINOR 1
#define SPARSEMAT_VERSION_PATCH 0

/// Single comparable integer, e.g. `#if SPARSEMAT_VERSION >= SPARSEMAT_VERSION_CHECK(0, 2, 0)`.
#define SPARSEMAT_VERSION_CHECK(major, minor, patch) (((major) * 10000) + ((minor) * 100) + (patch))

#define SPARSEMAT_VERSION \
  SPARSEMAT_VERSION_CHECK(SPARSEMAT_VERSION_MAJOR, SPARSEMAT_VERSION_MINOR, SPARSEMAT_VERSION_PATCH)

#define SPARSEMAT_VERSION_STRING "0.1.0"
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
