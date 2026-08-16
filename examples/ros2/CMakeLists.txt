cmake_minimum_required(VERSION 3.20)
project(cadence_ros2_example LANGUAGES CXX)

# Deliberately not reachable from the cadence root CMakeLists and not in CI. A
# GitHub runner has no ROS 2, and wiring an ament package into a header-only
# library's build would make every consumer of that library pay attention to a
# dependency they do not have. Build this one with colcon; see README.md.

# The default has to be chosen before the language is enabled, not after:
# enable_language(CUDA) defines CMAKE_CUDA_ARCHITECTURES itself, and a
# `if(NOT DEFINED ...)` guard placed after it silently never fires, leaving the
# node compiled for a decade-old architecture on whatever card you actually have.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
endif()
enable_language(CUDA)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

# cadence is header-only, so consuming it is an include path and C++17. An
# installed package wins; failing that this falls back to the tree the example
# ships inside, so it builds whether or not you ran `cmake --install` first.
find_package(cadence QUIET)
if(NOT cadence_FOUND)
  add_library(cadence_headers INTERFACE)
  target_include_directories(cadence_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
  target_compile_features(cadence_headers INTERFACE cxx_std_17)
  target_link_libraries(cadence_headers INTERFACE Threads::Threads)
  add_library(cadence::cadence ALIAS cadence_headers)
  message(STATUS "cadence_ros2_example: using the in-tree cadence headers")
endif()

add_executable(perception_node src/perception_node.cu)
set_target_properties(perception_node PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
# Imported targets rather than ament_target_dependencies, which links in the
# plain signature and so cannot be mixed with the PRIVATE form cadence needs.
target_link_libraries(perception_node PRIVATE
  cadence::cadence
  rclcpp::rclcpp
  ${std_msgs_TARGETS})

install(TARGETS perception_node DESTINATION lib/${PROJECT_NAME})

ament_package()
