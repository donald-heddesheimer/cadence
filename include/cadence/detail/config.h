// cadence — configuration surface.
//
// The struct is the source of truth; environment variables override it so a
// run can be re-pointed without a rebuild. Precedence, lowest to highest:
//   defaults  <  cadence::Configure(cfg)  <  environment
#pragma once

#include <cstdlib>
#include <string>

namespace cadence {

struct Config {
  // Iterations discarded per label before statistics accumulate. The first launches pay for context creation, JIT, and library autotuning (cuBLAS/cuDNN), so counting them poisons the mean and the minimum.
  unsigned warmupIterations = 3;

  // Where Report() writes. Empty disables file output.
  std::string outputPath = "cadence.csv";

  // Mirror every scope as an NVTX range so the same instrumentation lights up Nsight Systems when a tool is attached. Costs ~nothing when it is not.
  bool nvtxEnabled = true;

  // Master runtime gate. Distinct from -DCADENCE_DISABLE, which removes the code entirely; this one is for flipping instrumentation off in a build that still contains it.
  bool enabled = true;

  // Flush and write on static destruction. Explicit Report() is more reliable because it runs before the CUDA runtime tears itself down.
  bool writeOnExit = true;
};

namespace detail {

inline const char* EnvOrNull(const char* name) {
  const char* value = std::getenv(name);
  return (value && *value) ? value : nullptr;
}

inline bool ParseBool(const char* value, bool fallback) {
  if (!value) return fallback;
  const std::string text(value);
  if (text == "0" || text == "false" || text == "off" || text == "no") return false;
  if (text == "1" || text == "true" || text == "on" || text == "yes") return true;
  return fallback;
}

// CADENCE_WARMUP, CADENCE_OUTPUT, CADENCE_NVTX, CADENCE_ENABLE.
inline void ApplyEnvironmentOverrides(Config& config) {
  if (const char* warmup = EnvOrNull("CADENCE_WARMUP")) {
    config.warmupIterations = static_cast<unsigned>(std::strtoul(warmup, nullptr, 10));
  }
  if (const char* output = EnvOrNull("CADENCE_OUTPUT")) {
    config.outputPath = output;
  }
  config.nvtxEnabled = ParseBool(EnvOrNull("CADENCE_NVTX"), config.nvtxEnabled);
  config.enabled = ParseBool(EnvOrNull("CADENCE_ENABLE"), config.enabled);
}

}  // namespace detail
}  // namespace cadence
