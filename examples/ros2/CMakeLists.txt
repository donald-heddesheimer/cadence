cmake_minimum_required(VERSION 3.20)
project(cadence_ros2_example LANGUAGES CXX)

# This standalone ament package builds with colcon; see README.md.

# Set the default before enable_language(CUDA), which defines the variable.
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

# Prefer an installed cadence package and fall back to the in-tree headers.
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
# Imported targets keep target_link_libraries on its keyword signature.
target_link_libraries(perception_node PRIVATE
  cadence::cadence
  rclcpp::rclcpp
  ${std_msgs_TARGETS})

install(TARGETS perception_node DESTINATION lib/${PROJECT_NAME})

ament_package()
