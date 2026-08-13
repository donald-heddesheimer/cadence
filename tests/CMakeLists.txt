# Host-only tests: no CUDA runtime, no GPU, no driver. They cover the parts of
# cadence that decide whether the numbers are right -- statistics, warmup
# discard, interning, CSV shape, the disabled build -- so CI can run them
# anywhere.
add_executable(cadence_host_tests host_tests.cc)
target_link_libraries(cadence_host_tests PRIVATE cadence::cadence)
target_compile_definitions(cadence_host_tests PRIVATE CADENCE_HAS_CUDA=0)
add_test(NAME cadence_host_tests COMMAND cadence_host_tests)

# The same header set compiled with the master switch off. Proves the macros
# vanish rather than degrading into runtime branches.
add_executable(cadence_disabled_tests disabled_tests.cc)
target_link_libraries(cadence_disabled_tests PRIVATE cadence::cadence)
target_compile_definitions(cadence_disabled_tests PRIVATE CADENCE_HAS_CUDA=0 CADENCE_DISABLE)
add_test(NAME cadence_disabled_tests COMMAND cadence_disabled_tests)

# Every header has to compile on its own. A header-only library that only works
# when you include its headers in one particular order is a trap, and this is
# the cheapest possible way to never ship one.
file(GLOB CADENCE_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include/cadence/*.h"
  "${CMAKE_SOURCE_DIR}/include/cadence/detail/*.h")
set(CADENCE_SELF_CONTAINED_SOURCES "")
foreach(header ${CADENCE_HEADERS})
  string(REGEX REPLACE "[/.]" "_" stem "${header}")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/self_contained_${stem}.cc")
  file(WRITE "${generated}" "#include <${header}>\n")
  list(APPEND CADENCE_SELF_CONTAINED_SOURCES "${generated}")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/self_contained_main.cc" "int main() { return 0; }\n")
add_executable(cadence_self_contained
  ${CADENCE_SELF_CONTAINED_SOURCES}
  "${CMAKE_CURRENT_BINARY_DIR}/self_contained_main.cc")
target_link_libraries(cadence_self_contained PRIVATE cadence::cadence)
add_test(NAME cadence_self_contained COMMAND cadence_self_contained)

# Device tests need a GPU as well as a compiler, so they are built when nvcc is
# present and simply skip themselves at runtime when no device answers.
if(CADENCE_HAVE_CUDA)
  add_executable(cadence_device_tests device_tests.cu)
  target_link_libraries(cadence_device_tests PRIVATE cadence::cadence)
  set_target_properties(cadence_device_tests PROPERTIES CUDA_STANDARD 17)
  add_test(NAME cadence_device_tests COMMAND cadence_device_tests)
  set_tests_properties(cadence_device_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
