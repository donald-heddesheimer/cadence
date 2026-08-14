// cadence: the deferred-flush registry.
//
// A profiler that serializes the pipeline is measuring itself.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cadence/detail/config.h"
#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/report.h"
#include "cadence/detail/stats.h"
#include "cadence/detail/thread_state.h"

#if CADENCE_HAS_CUDA
#include "cadence/detail/event_pool.h"
#endif

namespace cadence {
    namespace detail {

    // Accumulated samples for one label, across every flush so far. Indexed by LabelId.
    struct LabelSamples {
        std::vector<double> deviceMs;
        std::vector<double> hostMs;   // Host scopes, or CPU-issue time for device scopes.
        std::uint64_t seen = 0;       // Observations kept for statistics, warmup included.
        std::uint64_t discarded = 0;  // Observations dropped as warmup.
        bool hasDevice = false;
    };

    class Registry {
       public:
        static Registry& Instance() {
            // Leaked on purpose. Thread-local state, the atexit report and the CUDA runtime all unwind against each other at shutdown, and a registry that cannot be destroyed cannot be used after destruction.
            static Registry* instance = new Registry();
            return *instance;
        }

        Registry(const Registry&) = delete;
        Registry& operator=(const Registry&) = delete;

        Config GetConfig() const {
            std::lock_guard<std::mutex> lock(configMutex_);
            return config_;
        }

        void Configure(const Config& config) {
            std::lock_guard<std::mutex> lock(configMutex_);
            config_ = config;
            ApplyEnvironmentOverrides(config_);
            PublishHotConfig(config_);
        }

        bool IsEnabled() const { return hotConfig.enabled.load(std::memory_order_relaxed); }

        std::uint64_t FlushGeneration() const { return flushGeneration_.load(std::memory_order_relaxed); }

#if CADENCE_HAS_CUDA
        // Tops up a thread's private event cache. One lock per NUM_EVENTS_PER_REFILL scopes rather than two per scope.
        void RefillEventCache(std::vector<cudaEvent_t>& cache) {
            std::lock_guard<std::mutex> lock(poolMutex_);
            eventPool_.AcquireInto(cache, NUM_EVENTS_PER_REFILL);
        }

        void ReturnEvents(std::vector<cudaEvent_t>& events) {
            if (events.empty()) return;
            std::lock_guard<std::mutex> lock(poolMutex_);
            eventPool_.ReleaseAll(events);
        }

        // CUDA events ever created and not destroyed. A steady-state loop should settle on a constant: a number that climbs means events are being taken and never handed back.
        std::size_t LiveEventCount() const {
            std::lock_guard<std::mutex> lock(poolMutex_);
            return eventPool_.LiveCount();
        }
#endif

        // Appends a host observation without a scope object. Interns on every call, so it is for tests and for callers who already have a duration in hand. 
        void RecordHost(const char* label, double elapsedMs);

        std::shared_ptr<ThreadState> RegisterThread() {
            auto state = std::make_shared<ThreadState>();
            std::lock_guard<std::mutex> lock(threadsMutex_);
            threads_.push_back(state);
            return state;
        }

        // Consumes every pending record from every thread. 
        void Flush() {
            // Bumping first means a thread that starts a stage after this point opens a fresh chain rather than extending one whose events are about to be recycled.
            flushGeneration_.fetch_add(1, std::memory_order_relaxed);

            std::vector<std::shared_ptr<ThreadState>> blocks;
            {
                std::lock_guard<std::mutex> lock(threadsMutex_);
                blocks = threads_;
            }

            std::vector<HostRecord> hostBatch;
#if CADENCE_HAS_CUDA
            // Kept per thread rather than concatenated: within one thread the records for a stream are in program order, hence in stream order, and that is what makes it sound to wait on only the last of them.
            std::vector<std::vector<DeviceRecord>> deviceBatches;
            deviceBatches.reserve(blocks.size());
#endif
            for (const std::shared_ptr<ThreadState>& block : blocks) {
                std::lock_guard<std::mutex> lock(block->mutex);
                if (!block->pendingHost.empty()) {
                    hostBatch.insert(hostBatch.end(), block->pendingHost.begin(), block->pendingHost.end());
                    block->pendingHost.clear();
                }
#if CADENCE_HAS_CUDA
                if (!block->pendingDevice.empty()) {
                    deviceBatches.emplace_back();
                    deviceBatches.back().swap(block->pendingDevice);
                    block->pendingDevice.reserve(NUM_RECORDS_RESERVED);
                }
#endif
            }

#if CADENCE_HAS_CUDA
            // Events on a stream complete in the order they were recorded, so waiting on the last stop event of a stream implies every earlier one on that stream. The naive loop calls cudaEventSynchronize once per record; on an already-complete event that still costs ~140 ns of driver round trip, which for a five-stage iteration is most of the flush.
            std::vector<cudaEvent_t> waits;
            std::vector<cudaStream_t> seenStreams;
            for (const std::vector<DeviceRecord>& batch : deviceBatches) {
                seenStreams.clear();
                for (std::size_t i = batch.size(); i > 0; --i) {
                    const DeviceRecord& record = batch[i - 1];
                    if (!record.stop) continue;
                    bool seen = false;
                    for (cudaStream_t stream : seenStreams) {
                        if (stream == record.stream) {
                            seen = true;
                            break;
                        }
                    }
                    if (seen) continue;
                    seenStreams.push_back(record.stream);
                    waits.push_back(record.stop);
                }
            }
            for (cudaEvent_t event : waits) cudaEventSynchronize(event);

            std::vector<cudaEvent_t> spent;
#endif

            // Read before taking samplesMutex_, so the only lock this function ever holds two of at once is none.
            const unsigned warmup = WarmupIterations();
            {
                std::lock_guard<std::mutex> lock(samplesMutex_);
#if CADENCE_HAS_CUDA
                for (const std::vector<DeviceRecord>& batch : deviceBatches) {
                    for (const DeviceRecord& record : batch) {
                        float elapsedMs = 0.0f;
                        const bool valid =
                            record.start && record.stop &&
                            cudaEventElapsedTime(&elapsedMs, record.start, record.stop) == cudaSuccess;
                        if (valid) {
                            LabelSamples& samples = SamplesFor(record.label);
                            samples.hasDevice = true;
                            if (KeepSample(samples, warmup)) {
                                samples.deviceMs.push_back(static_cast<double>(elapsedMs));
                                // The GPU figure came from CUDA events and stands on its own; only the host half is dropped when the clock did not move, so a stalled clock costs you the issue-time row and nothing else.
                                if (record.hostIssueNs > 0) {
                                    samples.hostMs.push_back(static_cast<double>(record.hostIssueNs) * 1e-6);
                                } else {
                                    ++stalledClockRecords_;
                                }
                            }
                        } else if (record.start || record.stop) {
                            ++failedRecords_;
                        }
                        if (record.ownsStart) spent.push_back(record.start);
                        spent.push_back(record.stop);
                    }
                }
#endif
                // A span of exactly zero nanoseconds is not a fast measurement, it is a failed one: two steady_clock reads with real work between them cannot return the same value. It happens on virtualized hosts, where CLOCK_MONOTONIC can hand back a stale value after a blocking call, and one such sample is enough to pin a label's minimum at zero and stretch its jitter and histogram across a range nothing ever occupied. Dropping it and saying so is more honest than reporting a number the machine did not measure.
                for (const HostRecord& record : hostBatch) {
                    LabelSamples& samples = SamplesFor(record.label);
                    if (KeepSample(samples, warmup)) {
                        if (record.elapsedNs > 0) {
                            samples.hostMs.push_back(static_cast<double>(record.elapsedNs) * 1e-6);
                        } else {
                            ++stalledClockRecords_;
                        }
                    }
                }
            }

#if CADENCE_HAS_CUDA
            ReturnEvents(spent);
#endif
            PruneRetiredThreads();
        }

        // Flush() first if you want the current loop's records included.
        std::vector<Stats> Snapshot() const {
            const std::vector<std::string> names = LabelTable::Instance().Names();
            // Read before samplesMutex_ so this function never holds two locks at once.
            const Config config = GetConfig();

            std::lock_guard<std::mutex> lock(samplesMutex_);
            const BudgetTarget target = ResolveBudgetTarget(config, names);

            std::vector<Stats> results;
            results.reserve(samples_.size());
            for (std::size_t id = 0; id < samples_.size(); ++id) {
                const LabelSamples& samples = samples_[id];
                if (samples.deviceMs.empty() && samples.hostMs.empty()) continue;
                const std::string& name = id < names.size() ? names[id] : std::string();
                if (samples.hasDevice) {
                    const double budget = target.Matches(id, ScopeKind::Device) ? config.budgetMs : 0.0;
                    results.push_back(ComputeStats(name, ScopeKind::Device, samples.deviceMs, samples.discarded, budget));
                }
                if (!samples.hostMs.empty()) {
                    // For a device scope this row is the CPU-issue side of the same label: compare it against the device row to see launch-bound vs compute-bound.
                    const double budget = target.Matches(id, ScopeKind::Host) ? config.budgetMs : 0.0;
                    results.push_back(ComputeStats(name, ScopeKind::Host, samples.hostMs, samples.discarded, budget));
                }
            }
            return results;
        }

        void Reset() {
            std::lock_guard<std::mutex> lock(samplesMutex_);
            samples_.clear();
            failedRecords_ = 0;
            stalledClockRecords_ = 0;
        }

        std::size_t FailedRecordCount() const {
            std::lock_guard<std::mutex> lock(samplesMutex_);
            return failedRecords_;
        }

        // Host spans discarded because the clock did not advance across them. Nonzero means the machine's monotonic clock is unreliable under load, not that the code being measured was fast.
        std::size_t StalledClockCount() const {
            std::lock_guard<std::mutex> lock(samplesMutex_);
            return stalledClockRecords_;
        }

        // Suppresses the exit-time write: the application reported explicitly, and that report was taken while the CUDA runtime was certainly still alive.
        void MarkReported() {
            std::lock_guard<std::mutex> lock(configMutex_);
            reported_ = true;
        }

        void WriteTo(std::ostream& out) const {
            const Config configCopy = GetConfig();
            WriteReport(out, configCopy, QueryRunInfo(), Snapshot(), FailedRecordCount(), StalledClockCount());
        }

       private:
        Registry() {
            ApplyEnvironmentOverrides(config_);
            PublishHotConfig(config_);
            // Last-resort output for applications that never call Report(). It runs before this object would be destroyed -- which it never is -- because atexit handlers and static destructors unwind in one interleaved reverse-order sequence and this registration happens after construction.
            std::atexit(&Registry::AtExitHandler);
        }

        static void AtExitHandler() {
            Registry& registry = Instance();
            const Config config = registry.GetConfig();
            if (!config.writeOnExit) return;
            {
                std::lock_guard<std::mutex> lock(registry.configMutex_);
                if (registry.reported_) return;
            }
            // The CUDA runtime may already be shutting down; Flush() tolerates that and the report simply loses whatever was still pending.
            registry.Flush();
            if (config.reportStream) registry.WriteTo(*config.reportStream);
            if (!config.outputPath.empty()) {
                std::ofstream out(config.outputPath);
                if (out) registry.WriteTo(out);
            }
        }

        unsigned WarmupIterations() const {
            std::lock_guard<std::mutex> lock(configMutex_);
            return config_.warmupIterations;
        }

        // Exactly one row in the report carries the budget, so the target is resolved once per snapshot rather than tested per row.
        struct BudgetTarget {
            bool active = false;
            std::size_t id = 0;
            ScopeKind kind = ScopeKind::Host;

            bool Matches(std::size_t candidate, ScopeKind candidateKind) const {
                return active && candidate == id && candidateKind == kind;
            }
        };

        // Called with samplesMutex_ held. An explicit label wins; a named label with GPU work is held to its GPU time, since that is what someone naming a kernel means. With no label named, the budget falls to the loop span: the sole label that recorded host time and never launched anything, which is the CADENCE_SCOPE around the iteration. Ambiguity resolves to no budget rather than to a guess, because a deadline reported against the wrong row is worse than no deadline at all.
        BudgetTarget ResolveBudgetTarget(const Config& config, const std::vector<std::string>& names) const {
            BudgetTarget target;
            if (config.budgetMs <= 0.0) return target;

            if (!config.budgetLabel.empty()) {
                for (std::size_t id = 0; id < samples_.size(); ++id) {
                    if (id >= names.size() || names[id] != config.budgetLabel) continue;
                    const LabelSamples& samples = samples_[id];
                    if (samples.hasDevice) {
                        target = BudgetTarget{true, id, ScopeKind::Device};
                    } else if (!samples.hostMs.empty()) {
                        target = BudgetTarget{true, id, ScopeKind::Host};
                    }
                    return target;
                }
                return target;
            }

            std::size_t hostOnlyCount = 0;
            for (std::size_t id = 0; id < samples_.size(); ++id) {
                const LabelSamples& samples = samples_[id];
                if (samples.hasDevice || samples.hostMs.empty()) continue;
                ++hostOnlyCount;
                target = BudgetTarget{true, id, ScopeKind::Host};
            }
            if (hostOnlyCount != 1) target.active = false;
            return target;
        }

        // Called with samplesMutex_ held.
        LabelSamples& SamplesFor(LabelId id) {
            if (id >= samples_.size()) samples_.resize(id + 1);
            return samples_[id];
        }

        // Warmup discard. Called with samplesMutex_ held; increments the per-label counter and reports whether this observation should contribute to statistics.
        bool KeepSample(LabelSamples& samples, unsigned warmup) {
            const std::uint64_t index = samples.seen++;
            if (index < warmup) {
                samples.discarded = index + 1;
                return false;
            }
            return true;
        }

        void PruneRetiredThreads() {
            std::lock_guard<std::mutex> lock(threadsMutex_);
            std::size_t kept = 0;
            for (std::size_t i = 0; i < threads_.size(); ++i) {
                const std::shared_ptr<ThreadState>& state = threads_[i];
                bool drop = false;
                {
                    std::lock_guard<std::mutex> stateLock(state->mutex);
                    drop = state->retired && state->pendingHost.empty();
#if CADENCE_HAS_CUDA
                    drop = drop && state->pendingDevice.empty();
#endif
                }
                if (!drop) threads_[kept++] = threads_[i];
            }
            threads_.resize(kept);
        }

        mutable std::mutex configMutex_;
        Config config_;
        bool reported_ = false;

        mutable std::mutex samplesMutex_;
        std::vector<LabelSamples> samples_;
        std::size_t failedRecords_ = 0;
        std::size_t stalledClockRecords_ = 0;

        std::mutex threadsMutex_;
        std::vector<std::shared_ptr<ThreadState>> threads_;

        std::atomic<std::uint64_t> flushGeneration_{1};

#if CADENCE_HAS_CUDA
        mutable std::mutex poolMutex_;
        EventPool eventPool_;
#endif
    };

    // Holds the calling thread's block alive and marks it retired on the way out. Retirement is a flag rather than a removal because Flush() may be reading the block at that moment; the registry drops it at the next flush that finds it drained.
    class ThreadStateHandle {
       public:
        ThreadStateHandle() : state_(Registry::Instance().RegisterThread()) {}

        ~ThreadStateHandle() {
#if CADENCE_HAS_CUDA
            Registry::Instance().ReturnEvents(state_->eventCache);
#endif
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->retired = true;
        }

        ThreadState& Get() const { return *state_; }

       private:
        std::shared_ptr<ThreadState> state_;
    };

    CADENCE_ALWAYS_INLINE ThreadState& TlsState() {
        static thread_local ThreadStateHandle handle;
        return handle.Get();
    }

    // Defined out of line because it needs TlsState(), which needs Registry to be a complete type.
    inline void Registry::RecordHost(const char* label, double elapsedMs) {
        const LabelHandle handle = LabelTable::Instance().Intern(label);
        ThreadState& state = TlsState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.pendingHost.push_back(HostRecord{handle.id, static_cast<std::int64_t>(elapsedMs * 1e6)});
    }

    // One observation in every `sampleEvery`. The counter lives in the label handle, so this costs an increment on a cache line only this label touches; the common case is sampleEvery == 1 and a branch the predictor gets right every time.
    CADENCE_ALWAYS_INLINE bool ShouldSample(const LabelHandle& handle) {
        const unsigned every = hotConfig.sampleEvery.load(std::memory_order_relaxed);
        if (CADENCE_LIKELY(every <= 1)) return true;
        if (!handle.observations) return true;
        const std::uint64_t index = handle.observations->fetch_add(1, std::memory_order_relaxed);
        return (index % every) == 0;
    }

#if CADENCE_HAS_CUDA
    // Pops one event from the thread's private cache, refilling from the shared pool when it runs dry. Returns nullptr only if the runtime will not create events, which callers treat as "this scope goes unmeasured" rather than as fatal.
    CADENCE_ALWAYS_INLINE cudaEvent_t TakeEvent(ThreadState& state) {
        if (CADENCE_UNLIKELY(state.eventCache.empty())) {
            Registry::Instance().RefillEventCache(state.eventCache);
            if (state.eventCache.empty()) return nullptr;
        }
        cudaEvent_t event = state.eventCache.back();
        state.eventCache.pop_back();
        return event;
    }
#endif

    }  // namespace detail
}  // namespace cadence
