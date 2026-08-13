// cadence: configuration surface.
#pragma once

#include <atomic>
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

        // Measure one observation in every N per label; 1 measures all of them.
        unsigned sampleEvery = 1;
    };

    namespace detail {
    // Hot-path mirrors of the settings a scope constructor consults.
    struct HotConfig {
        std::atomic<bool> enabled{true};
        std::atomic<bool> nvtxEnabled{true};
        std::atomic<unsigned> sampleEvery{1};
    };

    inline HotConfig hotConfig;

    inline void PublishHotConfig(const Config& config) {
        hotConfig.enabled.store(config.enabled, std::memory_order_relaxed);
        hotConfig.nvtxEnabled.store(config.nvtxEnabled, std::memory_order_relaxed);
        hotConfig.sampleEvery.store(config.sampleEvery < 1 ? 1 : config.sampleEvery, std::memory_order_relaxed);
    }

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

    // CADENCE_WARMUP, CADENCE_OUTPUT, CADENCE_NVTX, CADENCE_ENABLE, CADENCE_SAMPLE.
    inline void ApplyEnvironmentOverrides(Config& config) {
        if (const char* warmup = EnvOrNull("CADENCE_WARMUP")) {
            config.warmupIterations = static_cast<unsigned>(std::strtoul(warmup, nullptr, 10));
        }
        if (const char* output = EnvOrNull("CADENCE_OUTPUT")) {
            config.outputPath = output;
        }
        if (const char* sample = EnvOrNull("CADENCE_SAMPLE")) {
            const unsigned parsed = static_cast<unsigned>(std::strtoul(sample, nullptr, 10));
            config.sampleEvery = parsed < 1 ? 1 : parsed;
        }
        config.nvtxEnabled = ParseBool(EnvOrNull("CADENCE_NVTX"), config.nvtxEnabled);
        config.enabled = ParseBool(EnvOrNull("CADENCE_ENABLE"), config.enabled);
    }
    }  // namespace detail
}  // namespace cadence
