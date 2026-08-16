// cadence: configuration surface.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

namespace cadence {
    // Observations retained per label per row before the reservoir starts replacing rather than appending. A library meant to be left in a production loop cannot keep every observation: at 100 Hz across five labels a day-long run accumulates roughly 43 million doubles, and a profiler that grows without bound until the process is killed is not one you leave switched on. 32768 caps a row at 256 KB while still estimating a p95 to well within a percent.
    inline constexpr std::size_t NUM_SAMPLES_RETAINED = 32768;

    // Whether the report is printed with ANSI colour. Auto colours only when the report is going to a terminal, which keeps a redirected run and the outputPath copy as clean text; escape codes in a log file are worse than no colour at all.
    enum class ColorMode { Auto, Always, Never };

    struct Config {
        // Iterations discarded per label before statistics accumulate. The first launches pay for context creation, JIT, and library autotuning (cuBLAS/cuDNN), so counting them poisons the mean and the minimum.
        unsigned warmupIterations = 3;

        // Report() always renders to `reportStream` below. Setting this additionally writes the same text to a file, for a run whose console you will not be watching. Empty by default: the report is meant to be read where it is printed, not hunted for on disk.
        std::string outputPath;

        // Where the report is printed. stdout by default, the way a benchmark tool prints its results; point it at std::cerr to keep it out of a pipeline, or at nullptr to suppress printing and rely on outputPath alone.
        std::ostream* reportStream = &std::cout;

        // Colour in the report. The default colours a terminal and nothing else; NO_COLOR in the environment turns it off everywhere.
        ColorMode colorOutput = ColorMode::Auto;

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

        // Observations retained per label per row before new ones start replacing old ones at random rather than accumulating. Bounds a long run's memory: count, mean, stddev, min, max and the deadline verdict stay exact regardless, and only the percentiles and the histogram become estimates. Zero retains everything, which is what a short benchmark wants and what a loop left running for a week does not.
        std::size_t maxSamplesPerLabel = NUM_SAMPLES_RETAINED;

        // Slowest iterations kept whole, with every stage that ran inside them, and printed as a breakdown under the table. This is the question the summary raises and cannot answer: not how often the loop missed, but which passes missed and what was slow in them. Costs one small vector per retained iteration and no extra CUDA work. Zero turns the section off.
        std::size_t numWorstIterations = 3;

        // Where to write a Chrome Trace Event JSON file of those retained iterations, openable at https://ui.perfetto.dev. Empty means no trace, which is the default because placing GPU spans on the host timeline costs an extra elapsed-time query per record at flush.
        std::string tracePath;
    };

    namespace detail {
    // Hot-path mirrors of the settings a scope constructor consults.
    struct HotConfig {
        std::atomic<bool> enabled{true};
        std::atomic<bool> nvtxEnabled{true};
        std::atomic<unsigned> sampleEvery{1};
        // Read by Flush() rather than by a scope. Counting misses as observations arrive is what keeps the deadline verdict exact once the reservoir starts discarding samples; the cost is that a budget changed part way through a run is only applied to what came after it.
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
        // NO_COLOR is read first and CADENCE_COLOR second, so the library-specific setting can still turn colour back on. Honouring the convention at all is the point: a user who set NO_COLOR has already said what they want of every tool they run.
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
