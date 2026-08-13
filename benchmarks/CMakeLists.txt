# The overhead numbers in the README come from here. They are measured on the
# machine you run this on, which is the only machine whose numbers matter.
add_executable(cadence_overhead overhead.cu)
target_link_libraries(cadence_overhead PRIVATE cadence::cadence)
set_target_properties(cadence_overhead PROPERTIES CUDA_STANDARD 17)
