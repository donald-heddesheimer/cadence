// cadence configuration.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

namespace cadence {
    // Default reservoir size per report row (256 KiB of doubles).
    inline constexpr std::size_t NUM_SAMPLES_RETAINED = 32768;

    // ANSI color policy for report output.
    enum class ColorMode { Auto, Always, Never };

    struct Config {
        // Initial observations discarded per label.
        unsigned warmupIterations = 3;

        // Optional path for a copy of the text report.
        std::string outputPath;

        // Report destination; nullptr suppresses stream output.
        std::ostream* reportStream = &std::cout;

        // Color terminals automatically; NO_COLOR disables color.
        ColorMode colorOutput = ColorMode::Auto;

        // Use UTF-8 table and histogram glyphs; false selects ASCII.
        bool unicodeOutput = true;

        // Mirror scopes as NVTX ranges for attached profiling tools.
        bool nvtxEnabled = true;

        // Runtime instrumentation gate. -DCADENCE_DISABLE removes it at compile time.
        bool enabled = true;

        // Write a fallback report at process exit.
        bool writeOnExit = true;

        // Measure one observation in every N per label; 1 measures all of them.
        unsigned sampleEvery = 1;

        // Deadline in milliseconds; zero disables deadline evaluation.
        double budgetMs = 0.0;

        // Deadline label. Empty selects an unambiguous host-only loop scope.
        std::string budgetLabel;

        // Reservoir size per row. Zero retains all observations. Aggregates and
        // deadline counts remain exact when the reservoir is full.
        std::size_t maxSamplesPerLabel = NUM_SAMPLES_RETAINED;

        // Number of slow iterations retained with their stage breakdowns.
        std::size_t numWorstIterations = 3;

        // Optional Chrome Trace Event JSON path for retained iterations. Enabling
        // tracing adds an elapsed-time query per device record during flush.
        std::string tracePath;
    };

    namespace detail {
    // Hot-path mirrors of the settings a scope constructor consults.
    struct HotConfig {
        std::atomic<bool> enabled{true};
        std::atomic<bool> nvtxEnabled{true};
        std::atomic<unsigned> sampleEvery{1};
        // Read by Flush(); budget changes apply only to later observations.
        std::atomic<double> budgetMs{0.0};
        std::atomic<std::size_t> maxSamplesPerLabel{NUM_SAMPLES_RETAINED};
        std::atomic<std::size_t> numWorstIterations{3};
        // Whether flush should place spans on an absolute timeline. Only a trace needs that, and it is not free, so the flush path reads this rather than the path string.
        std::atomic<bool> traceEnabled{false};
    };

    inline HotConfig hotConfig;

    inline void PublishHotConfig(const Config& config) {
        hotConfig.enabled.store(config.enabled, std::memory_order_relaxed);
        hotConfig.nvtxEnabled.store(config.nvtxEnabled, std::memory_order_relaxed);
        hotConfig.sampleEvery.store(config.sampleEvery < 1 ? 1 : config.sampleEvery, std::memory_order_relaxed);
        hotConfig.budgetMs.store(config.budgetMs, std::memory_order_relaxed);
        hotConfig.maxSamplesPerLabel.store(config.maxSamplesPerLabel, std::memory_order_relaxed);
        hotConfig.numWorstIterations.store(config.numWorstIterations, std::memory_order_relaxed);
        hotConfig.traceEnabled.store(!config.tracePath.empty(), std::memory_order_relaxed);
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

    // CADENCE_WARMUP, CADENCE_OUTPUT, CADENCE_NVTX, CADENCE_ENABLE, CADENCE_SAMPLE, CADENCE_UNICODE, CADENCE_COLOR, NO_COLOR, CADENCE_BUDGET_MS, CADENCE_BUDGET_LABEL, CADENCE_MAX_SAMPLES, CADENCE_WORST, CADENCE_TRACE.
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
        if (const char* retained = EnvOrNull("CADENCE_MAX_SAMPLES")) {
            config.maxSamplesPerLabel = static_cast<std::size_t>(std::strtoull(retained, nullptr, 10));
        }
        if (const char* worst = EnvOrNull("CADENCE_WORST")) {
            config.numWorstIterations = static_cast<std::size_t>(std::strtoull(worst, nullptr, 10));
        }
        if (const char* trace = EnvOrNull("CADENCE_TRACE")) {
            config.tracePath = trace;
        }
        config.nvtxEnabled = ParseBool(EnvOrNull("CADENCE_NVTX"), config.nvtxEnabled);
        config.enabled = ParseBool(EnvOrNull("CADENCE_ENABLE"), config.enabled);
        config.unicodeOutput = ParseBool(EnvOrNull("CADENCE_UNICODE"), config.unicodeOutput);
        // CADENCE_COLOR takes precedence over the general NO_COLOR convention.
        if (EnvOrNull("NO_COLOR")) config.colorOutput = ColorMode::Never;
        if (const char* color = EnvOrNull("CADENCE_COLOR")) {
            if (std::string(color) == "auto") {
                config.colorOutput = ColorMode::Auto;
            } else {
                config.colorOutput = ParseBool(color, true) ? ColorMode::Always : ColorMode::Never;
            }
        }
        if (const char* budget = EnvOrNull("CADENCE_BUDGET_MS")) {
            config.budgetMs = std::strtod(budget, nullptr);
        }
        if (const char* label = EnvOrNull("CADENCE_BUDGET_LABEL")) {
            config.budgetLabel = label;
        }
    }
    }  // namespace detail
}  // namespace cadence
