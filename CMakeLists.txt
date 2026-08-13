cmake_minimum_required(VERSION 3.20)
project(cadence VERSION 0.1.0 LANGUAGES CXX)

# cadence is header-only. A consumer needs the include path and C++17 -- no
# compiled objects, no CUDA language enabled here, nothing to configure. That
# is the whole build.
add_library(cadence INTERFACE)
add_library(cadence::cadence ALIAS cadence)

target_compile_features(cadence INTERFACE cxx_std_17)
target_include_directories(cadence INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

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

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfig.cmake"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/cadenceTargets.cmake\")\n")

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cadenceConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cadence)
