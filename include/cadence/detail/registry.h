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
        SampleSet device;
        SampleSet host;               // Host scopes, or CPU-issue time for device scopes.
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
        // Tops up a thread's private event cache. One lock per NUM_EVENTS_PER_REFILL scopes rather than two per scope. The caller's current device must be `device`, since that is what cudaEventCreate binds a new event to.
        void RefillEventCache(int device, std::vector<cudaEvent_t>& cache) {
            std::lock_guard<std::mutex> lock(poolMutex_);
            eventPools_.For(device).AcquireInto(cache, NUM_EVENTS_PER_REFILL);
        }

        void ReturnEvents(int device, std::vector<cudaEvent_t>& events) {
            if (events.empty()) return;
            std::lock_guard<std::mutex> lock(poolMutex_);
            eventPools_.For(device).ReleaseAll(events);
        }

        // CUDA events ever created and not destroyed, across every device. A steady-state loop should settle on a constant: a number that climbs means events are being taken and never handed back.
        std::size_t LiveEventCount() const {
            std::lock_guard<std::mutex> lock(poolMutex_);
            return eventPools_.LiveCount();
        }

        // Scopes that recorded nothing because their stream was capturing into a CUDA graph. Read without the lock: it is a diagnostic count, and the report is rendered long after the loop that incremented it.
        void NoteCapturedScope() { capturedScopes_.fetch_add(1, std::memory_order_relaxed); }
        std::size_t CapturedScopeCount() const { return capturedScopes_.load(std::memory_order_relaxed); }
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

            // Grouped by the device that created them, because that is the only pool an event may be handed back to.
            std::vector<std::vector<cudaEvent_t>> spentByDevice;
#endif

            // Read before taking samplesMutex_, so the only lock this function ever holds two of at once is none.
            const unsigned warmup = WarmupIterations();
            const double budgetMs = hotConfig.budgetMs.load(std::memory_order_relaxed);
            const std::size_t capacity = hotConfig.maxSamplesPerLabel.load(std::memory_order_relaxed);
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
                                samples.device.Add(static_cast<double>(elapsedMs), budgetMs, capacity);
                                // The GPU figure came from CUDA events and stands on its own; only the host half is dropped when the clock did not move, so a stalled clock costs you the issue-time row and nothing else.
                                if (record.hostIssueNs > 0) {
                                    samples.host.Add(static_cast<double>(record.hostIssueNs) * 1e-6, budgetMs, capacity);
                                } else {
                                    ++stalledClockRecords_;
                                }
                            }
                        } else if (record.start || record.stop) {
                            ++failedRecords_;
                        }
                        const std::size_t index = record.device < 0 ? 0 : static_cast<std::size_t>(record.device);
                        if (index >= spentByDevice.size()) spentByDevice.resize(index + 1);
                        if (record.ownsStart && record.start) spentByDevice[index].push_back(record.start);
                        if (record.stop) spentByDevice[index].push_back(record.stop);
                    }
                }
#endif
                // A span of exactly zero nanoseconds is not a fast measurement, it is a failed one: two steady_clock reads with real work between them cannot return the same value. It happens on virtualized hosts, where CLOCK_MONOTONIC can hand back a stale value after a blocking call, and one such sample is enough to pin a label's minimum at zero and stretch its jitter and histogram across a range nothing ever occupied. Dropping it and saying so is more honest than reporting a number the machine did not measure.
                for (const HostRecord& record : hostBatch) {
                    LabelSamples& samples = SamplesFor(record.label);
                    if (KeepSample(samples, warmup)) {
                        if (record.elapsedNs > 0) {
                            samples.host.Add(static_cast<double>(record.elapsedNs) * 1e-6, budgetMs, capacity);
                        } else {
                            ++stalledClockRecords_;
                        }
                    }
                }
            }

#if CADENCE_HAS_CUDA
            for (std::size_t device = 0; device < spentByDevice.size(); ++device) {
                ReturnEvents(static_cast<int>(device), spentByDevice[device]);
            }
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
                if (samples.device.Empty() && samples.host.Empty()) continue;
                const std::string& name = id < names.size() ? names[id] : std::string();
                if (samples.hasDevice) {
                    const double budget = target.Matches(id, ScopeKind::Device) ? config.budgetMs : 0.0;
                    results.push_back(ComputeStatsFromSet(name, ScopeKind::Device, samples.device, samples.discarded, budget));
                }
                if (!samples.host.Empty()) {
                    // For a device scope this row is the CPU-issue side of the same label: compare it against the device row to see launch-bound vs compute-bound.
                    const double budget = target.Matches(id, ScopeKind::Host) ? config.budgetMs : 0.0;
                    results.push_back(ComputeStatsFromSet(name, ScopeKind::Host, samples.host, samples.discarded, budget));
                }
            }
            return results;
        }

        void Reset() {
#if CADENCE_HAS_CUDA
            capturedScopes_.store(0, std::memory_order_relaxed);
#endif
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
#if CADENCE_HAS_CUDA
            const std::size_t captured = CapturedScopeCount();
#else
            const std::size_t captured = 0;
#endif
            WriteReport(out, configCopy, QueryRunInfo(), Snapshot(), FailedRecordCount(), StalledClockCount(), captured);
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
                    } else if (!samples.host.Empty()) {
                        target = BudgetTarget{true, id, ScopeKind::Host};
                    }
                    return target;
                }
                return target;
            }

            std::size_t hostOnlyCount = 0;
            for (std::size_t id = 0; id < samples_.size(); ++id) {
                const LabelSamples& samples = samples_[id];
                if (samples.hasDevice || samples.host.Empty()) continue;
                ++hostOnlyCount;
                target = BudgetTarget{true, id, ScopeKind::Host};
            }
            if (hostOnlyCount != 1) target.active = false;
            return target;
        }

        // Called with samplesMutex_ held. Each set's reservoir is seeded from the label it belongs to, so which observations survive a capped run is reproducible from one run to the next.
        LabelSamples& SamplesFor(LabelId id) {
            if (id >= samples_.size()) {
                const std::size_t first = samples_.size();
                samples_.resize(id + 1);
                for (std::size_t i = first; i <= id; ++i) {
                    samples_[i].device.rngState = 0x9E3779B97F4A7C15ULL * (i + 1);
                    samples_[i].host.rngState = 0xBF58476D1CE4E5B9ULL * (i + 1);
                }
            }
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
        DeviceEventPools eventPools_;
        std::atomic<std::size_t> capturedScopes_{0};
#endif
    };

    // Holds the calling thread's block alive and marks it retired on the way out. Retirement is a flag rather than a removal because Flush() may be reading the block at that moment; the registry drops it at the next flush that finds it drained.
    class ThreadStateHandle {
       public:
        ThreadStateHandle() : state_(Registry::Instance().RegisterThread()) {}

        ~ThreadStateHandle() {
#if CADENCE_HAS_CUDA
            for (EventCache& cache : state_->eventCaches) {
                Registry::Instance().ReturnEvents(cache.device, cache.events);
            }
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
    // The device a scope's events must be created on, which is the one its kernel launch will go to. Measured at 20.6 ns on an RTX A4000, against 1490 ns for the cudaEventRecord it protects.
    CADENCE_ALWAYS_INLINE int CurrentDevice() {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 0;
        return device;
    }

    // True when nothing may be recorded on this stream because it is being captured into a CUDA graph.
    //
    // Recording anyway is not a matter of collecting a wrong number. cudaEventRecord succeeds during capture, but the event is baked into the graph rather than executed, and from then on every cudaEventSynchronize and cudaEventElapsedTime against it fails. Flushing while capture is still open poisons the capture outright: cudaStreamEndCapture then returns an error and hands the application back a null graph. Flushing after it closes leaves the graph intact but the events permanently unreadable, and recycling one into the pool hands an unrelated scope a handle that the instantiated graph silently re-records on every replay. Measured at 39.5 ns per check.
    CADENCE_ALWAYS_INLINE bool StreamIsCapturing(cudaStream_t stream) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (CADENCE_UNLIKELY(cudaStreamIsCapturing(stream, &status) != cudaSuccess)) {
            // The legacy default stream reports cudaErrorStreamCaptureImplicit while any other stream captures in global mode, so a failure here means capture is in play just as surely as a positive status does. Clear it rather than leaving our own probe as the error the application reads next.
            cudaGetLastError();
            return true;
        }
        return status != cudaStreamCaptureStatusNone;
    }

    // Pops one event from the thread's private cache for `device`, refilling from that device's pool when it runs dry. Returns nullptr only if the runtime will not create events, which callers treat as "this scope goes unmeasured" rather than as fatal.
    CADENCE_ALWAYS_INLINE cudaEvent_t TakeEvent(ThreadState& state, int device) {
        std::vector<cudaEvent_t>& cache = state.CacheFor(device);
        if (CADENCE_UNLIKELY(cache.empty())) {
            Registry::Instance().RefillEventCache(device, cache);
            if (cache.empty()) return nullptr;
        }
        cudaEvent_t event = cache.back();
        cache.pop_back();
        return event;
    }

    // Hands an unused event straight back to the thread's own cache. No lock, because the cache is private to this thread.
    CADENCE_ALWAYS_INLINE void GiveBackEvent(ThreadState& state, int device, cudaEvent_t event) {
        if (event) state.CacheFor(device).push_back(event);
    }

    // Sampling decision for a chained stage, made once per stream per flush generation and reused by every stage in that chain.
    //
    // ScopedStage borrows its predecessor's stop event, so the stages of one chain are not independent observations and cannot be thinned independently: drop the middle of a chain and the stage after the gap measures itself plus everything skipped before it. Deciding once per chain keeps every retained stage measuring exactly what it claims to, and makes sampleEvery mean "one iteration in N" for stages, which is what it already means for the loop those stages sit in.
    CADENCE_ALWAYS_INLINE bool ShouldSampleChain(const LabelHandle& label, cudaStream_t stream) {
        if (CADENCE_LIKELY(hotConfig.sampleEvery.load(std::memory_order_relaxed) <= 1)) return true;
        ThreadState& state = TlsState();
        StreamChain* chain = state.FindChain(stream);
        const std::uint64_t generation = Registry::Instance().FlushGeneration();
        if (chain->sampleGeneration != generation) {
            chain->sampleGeneration = generation;
            chain->sampled = ShouldSample(label);
        }
        return chain->sampled;
    }
#endif

    }  // namespace detail
}  // namespace cadence
