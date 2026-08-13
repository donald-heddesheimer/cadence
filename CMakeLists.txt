cmake_minimum_required(VERSION 3.20)
project(cadence VERSION 0.2.0 LANGUAGES CXX)

# cadence is header-only. A consumer needs the include path and C++17 -- no
# compiled objects, no CUDA language enabled here, nothing to configure. That
# is the whole build.
add_library(cadence INTERFACE)
add_library(cadence::cadence ALIAS cadence)

target_compile_features(cadence INTERFACE cxx_std_17)
target_include_directories(cadence INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
find_package(Threads REQUIRED)
target_link_libraries(cadence INTERFACE Threads::Threads)

# --- Developer tree -----------------------------------------------------------
# Tests, examples and benchmarks build only when cadence is the project being
# built. Someone who does add_subdirectory(cadence) in their own tree gets the
# headers and nothing else, which is the whole point of a header-only library.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CADENCE_IS_TOP_LEVEL ON)
else()
  set(CADENCE_IS_TOP_LEVEL OFF)
endif()

option(CADENCE_BUILD_TESTS "Build cadence tests" ${CADENCE_IS_TOP_LEVEL})
option(CADENCE_BUILD_EXAMPLES "Build cadence examples" ${CADENCE_IS_TOP_LEVEL})
option(CADENCE_BUILD_BENCHMARKS "Build cadence benchmarks" ${CADENCE_IS_TOP_LEVEL})

# The host tests, and everything CUDA-free, build anywhere. The device tests and
# examples need nvcc; point CMake at one with -DCMAKE_CUDA_COMPILER=... or the
# CUDACXX environment variable.
if(CADENCE_BUILD_TESTS OR CADENCE_BUILD_EXAMPLES OR CADENCE_BUILD_BENCHMARKS)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CADENCE_HAVE_CUDA ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 70)
      endif()
    endif()
  else()
    set(CADENCE_HAVE_CUDA OFF)
    message(STATUS "cadence: no CUDA compiler found; building the host-only targets")
  endif()
endif()

if(CADENCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(CADENCE_BUILD_EXAMPLES AND CADENCE_HAVE_CUDA)
  add_subdirectory(examples)
endif()
if(CADENCE_BUILD_BENCHMARKS AND CADENCE_HAVE_CUDA)
  add_subdirectory(benchmarks)
endif()

# --- Install ------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/cadence DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS cadence EXPORT cadenceTargets)
install(EXPORT cadenceTargets
  FILE cadenceTargets.cmake
  NAMESPACE cadence::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cadence)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT)

# find_package(cadence) has to pull in Threads itself: the exported target
# depends on it, and an imported target cannot carry that dependency for you.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfig.cmake"
"include(CMakeFindDependencyMacro)\n\
find_dependency(Threads)\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/cadenceTargets.cmake\")\n")

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cadence)
