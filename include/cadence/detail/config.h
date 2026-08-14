// cadence: configuration surface.
#pragma once

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

namespace cadence {
    struct Config {
        // Iterations discarded per label before statistics accumulate. The first launches pay for context creation, JIT, and library autotuning (cuBLAS/cuDNN), so counting them poisons the mean and the minimum.
        unsigned warmupIterations = 3;

        // Report() always renders to `reportStream` below. Setting this additionally writes the same text to a file, for a run whose console you will not be watching. Empty by default: the report is meant to be read where it is printed, not hunted for on disk.
        std::string outputPath;

        // Where the report is printed. stdout by default, the way a benchmark tool prints its results; point it at std::cerr to keep it out of a pipeline, or at nullptr to suppress printing and rely on outputPath alone.
        std::ostream* reportStream = &std::cout;

        // Box-drawing characters and block glyphs in the report. Turn it off for a terminal or a log collector that mangles UTF-8; the table then renders in pure ASCII.
        bool unicodeOutput = true;

        // Mirror every scope as an NVTX range so the same instrumentation lights up Nsight Systems when a tool is attached. Costs ~nothing when it is not.
        bool nvtxEnabled = true;

        // Master runtime gate. Distinct from -DCADENCE_DISABLE, which removes the code entirely; this one is for flipping instrumentation off in a build that still contains it.
        bool enabled = true;

        // Flush and write on static destruction. Explicit Report() is more reliable because it runs before the CUDA runtime tears itself down.
        bool writeOnExit = true;

        // Measure one observation in every N per label; 1 measures all of them.
        unsigned sampleEvery = 1;

        // The deadline one stage is held to, in milliseconds. Zero disables the check. A loop that must close at 100 Hz sets 10.0 and the report then says how often it did not, which is the question a mean cannot answer.
        double budgetMs = 0.0;

        // Which label the budget applies to. Empty picks the loop span automatically: the one label that recorded host time but never launched a kernel, which is the CADENCE_SCOPE wrapped around the iteration. Naming a label explicitly holds that label instead, preferring its GPU row when it has one.
        std::string budgetLabel;
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

    // CADENCE_WARMUP, CADENCE_OUTPUT, CADENCE_NVTX, CADENCE_ENABLE, CADENCE_SAMPLE, CADENCE_UNICODE, CADENCE_BUDGET_MS, CADENCE_BUDGET_LABEL.
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
        config.unicodeOutput = ParseBool(EnvOrNull("CADENCE_UNICODE"), config.unicodeOutput);
        if (const char* budget = EnvOrNull("CADENCE_BUDGET_MS")) {
            config.budgetMs = std::strtod(budget, nullptr);
        }
        if (const char* label = EnvOrNull("CADENCE_BUDGET_LABEL")) {
            config.budgetLabel = label;
        }
    }
    }  // namespace detail
}  // namespace cadence
