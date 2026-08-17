// cadence deferred-record registry.
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
#include "cadence/detail/trace.h"

#if CADENCE_HAS_CUDA
#include "cadence/detail/event_pool.h"
#endif

namespace cadence {
    namespace detail {

    // Keep execution and issue-time samples separate for each GPU.
    struct DeviceSamples {
        int device = 0;
        SampleSet gpu;    // GPU execution, measured with CUDA events.
        SampleSet issue;  // CPU time spent issuing it.
    };

    // Accumulated samples for one label, across every flush so far. Indexed by LabelId.
    struct LabelSamples {
        std::vector<DeviceSamples> devices;  // Ordered by device id; empty until a device scope records.
        SampleSet host;               // Plain host scopes, which belong to no GPU.
        std::uint64_t seen = 0;       // Observations kept for statistics, warmup included.
        std::uint64_t discarded = 0;  // Observations dropped as warmup.
        bool hasDevice = false;       // Set as soon as a device record arrives, warmup included, so a label does not read as host-only while its first iterations are being discarded.
    };

    // Create device slots on demand and keep them ordered for stable reports.
    // This free function remains available to host-only tests.
    inline DeviceSamples& SlotFor(LabelSamples& samples, LabelId label, int device) {
        std::size_t index = 0;
        while (index < samples.devices.size() && samples.devices[index].device < device) ++index;
        if (index == samples.devices.size() || samples.devices[index].device != device) {
            DeviceSamples slot;
            slot.device = device;
            // Seeded from the label and the device together, so which observations a capped run keeps is reproducible from one run to the next and two cards do not thin their reservoirs in lockstep.
            const std::uint64_t seed = (static_cast<std::uint64_t>(label) + 1) * 131u + static_cast<std::uint64_t>(device) + 1u;
            slot.gpu.rngState = 0x9E3779B97F4A7C15ULL * seed;
            slot.issue.rngState = 0xBF58476D1CE4E5B9ULL * seed;
            samples.devices.insert(samples.devices.begin() + static_cast<std::ptrdiff_t>(index), std::move(slot));
        }
        return samples.devices[index];
    }

#if CADENCE_HAS_CUDA
    // CUDA events must be created on the device that records them.
    CADENCE_ALWAYS_INLINE int CurrentDevice() {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 0;
        return device;
    }

    // Ties the GPU clock to the host clock for one flush on one device.
    //
    // Map a dedicated CUDA event to a host timestamp. Reusing an earlier stop
    // event would shift spans by the unbounded delay before Flush().
    struct TraceAnchor {
        cudaEvent_t event = nullptr;
        std::int64_t hostNs = 0;
        bool valid = false;
    };
#endif

    class Registry {
       public:
        static Registry& Instance() {
            // Intentionally process-lifetime to avoid static destruction order
            // conflicts with thread-local state and the CUDA runtime.
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
            // Preserve per-thread stream order for batched synchronization.
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
            // Synchronize only the final event recorded on each stream.
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
            const std::size_t numWorst = hotConfig.numWorstIterations.load(std::memory_order_relaxed);

            // A flush defines one iteration. Reuse the span buffer to avoid
            // steady-state allocations.
            const std::uint64_t iterationIndex = flushGeneration_.load(std::memory_order_relaxed) - 1;

#if CADENCE_HAS_CUDA
            // A host span already knows where it sits on the clock; only GPU spans need placing, so this is read on the device path alone.
            const bool tracing = hotConfig.traceEnabled.load(std::memory_order_relaxed);
            // Trace anchors map device durations onto the host timeline.
            std::vector<TraceAnchor> anchors;
            if (tracing) anchors = RecordTraceAnchors(deviceBatches);
#endif
            {
                std::lock_guard<std::mutex> lock(samplesMutex_);
                iterationSpans_.clear();
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
                                // The device is already on the record, so keying by it costs a scan over a handful of slots and nothing on the hot path.
                                DeviceSamples& slot = SlotFor(samples, record.label, record.device < 0 ? 0 : record.device);
                                slot.gpu.Add(static_cast<double>(elapsedMs), budgetMs, capacity);
                                // The GPU figure came from CUDA events and stands on its own; only the host half is dropped when the clock did not move, so a stalled clock costs you the issue-time row and nothing else.
                                if (record.hostIssueNs > 0) {
                                    slot.issue.Add(static_cast<double>(record.hostIssueNs) * 1e-6, budgetMs, capacity);
                                } else {
                                    ++stalledClockRecords_;
                                }
                                if (numWorst > 0) {
                                    IterationSpan span;
                                    span.label = record.label;
                                    span.kind = ScopeKind::Device;
                                    span.durationMs = static_cast<double>(elapsedMs);
                                    span.lane = LaneFor(record.stream);
                                    const std::size_t deviceIndex = record.device < 0 ? 0 : static_cast<std::size_t>(record.device);
                                    if (tracing && deviceIndex < anchors.size() && anchors[deviceIndex].valid) {
                                        float msBeforeAnchor = 0.0f;
                                        if (cudaEventElapsedTime(&msBeforeAnchor, record.start, anchors[deviceIndex].event) == cudaSuccess) {
                                            span.startNs = anchors[deviceIndex].hostNs - static_cast<std::int64_t>(static_cast<double>(msBeforeAnchor) * 1e6);
                                        }
                                    }
                                    iterationSpans_.push_back(span);
                                    // Include issue spans only in the timeline.
                                    if (tracing && record.hostIssueNs > 0) {
                                        IterationSpan issue;
                                        issue.label = record.label;
                                        issue.kind = ScopeKind::Host;
                                        issue.durationMs = static_cast<double>(record.hostIssueNs) * 1e-6;
                                        issue.startNs = record.hostStartNs;
                                        issue.lane = 0;
                                        iterationSpans_.push_back(issue);
                                    }
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
                // A zero-length host span indicates a stalled monotonic clock.
                for (const HostRecord& record : hostBatch) {
                    LabelSamples& samples = SamplesFor(record.label);
                    if (KeepSample(samples, warmup)) {
                        if (record.elapsedNs > 0) {
                            samples.host.Add(static_cast<double>(record.elapsedNs) * 1e-6, budgetMs, capacity);
                            if (numWorst > 0) {
                                IterationSpan span;
                                span.label = record.label;
                                span.kind = ScopeKind::Host;
                                span.durationMs = static_cast<double>(record.elapsedNs) * 1e-6;
                                span.startNs = record.startNs;
                                span.lane = 0;
                                iterationSpans_.push_back(span);
                            }
                        } else {
                            ++stalledClockRecords_;
                        }
                    }
                }

                // Inside the lock, because this is where worst_ is written and Snapshot reads it from another thread.
                if (numWorst > 0 && !iterationSpans_.empty()) {
                    RetainIfWorst(iterationIndex, IterationSpanMs(iterationSpans_), numWorst);
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
                if (samples.devices.empty() && samples.host.Empty()) continue;
                const std::string& name = id < names.size() ? names[id] : std::string();
                // One pair of rows per GPU. With a single device -- which is nearly every run -- this emits exactly the two rows it always did.
                for (const DeviceSamples& slot : samples.devices) {
                    const double gpuBudget = target.Matches(id, ScopeKind::Device) ? config.budgetMs : 0.0;
                    Stats gpu = ComputeStatsFromSet(name, ScopeKind::Device, slot.gpu, samples.discarded, gpuBudget);
                    gpu.device = slot.device;
                    results.push_back(std::move(gpu));
                    if (slot.issue.Empty()) continue;
                    // The CPU-issue side of the same label: compare it against the device row above to see launch-bound vs compute-bound.
                    const double issueBudget = target.Matches(id, ScopeKind::Host) ? config.budgetMs : 0.0;
                    Stats issue = ComputeStatsFromSet(name, ScopeKind::Host, slot.issue, samples.discarded, issueBudget);
                    issue.device = slot.device;
                    results.push_back(std::move(issue));
                }
                if (!samples.host.Empty()) {
                    // A plain CADENCE_SCOPE, which named no stream and so belongs to no GPU; its row keeps device = -1.
                    const double budget = target.Matches(id, ScopeKind::Host) ? config.budgetMs : 0.0;
                    results.push_back(ComputeStatsFromSet(name, ScopeKind::Host, samples.host, samples.discarded, budget));
                }
            }
            return results;
        }

        // The slowest iterations kept so far, slowest first, with their labels resolved. Ordered by the same measure the report ranks them by.
        std::vector<TraceIteration> WorstIterations() const {
            const std::vector<std::string> names = LabelTable::Instance().Names();
            std::lock_guard<std::mutex> lock(samplesMutex_);
            std::vector<TraceIteration> resolved;
            resolved.reserve(worst_.size());
            for (const IterationRecord& record : worst_) {
                TraceIteration iteration;
                iteration.index = record.index;
                iteration.spanMs = record.spanMs;
                iteration.spans.reserve(record.spans.size());
                for (const IterationSpan& span : record.spans) {
                    TraceSpan resolvedSpan;
                    resolvedSpan.label = span.label < names.size() ? names[span.label] : std::string();
                    resolvedSpan.kind = span.kind;
                    resolvedSpan.durationMs = span.durationMs;
                    resolvedSpan.startNs = span.startNs;
                    resolvedSpan.lane = span.lane;
                    iteration.spans.push_back(std::move(resolvedSpan));
                }
                resolved.push_back(std::move(iteration));
            }
            return resolved;
        }

        void Reset() {
#if CADENCE_HAS_CUDA
            capturedScopes_.store(0, std::memory_order_relaxed);
#endif
            std::lock_guard<std::mutex> lock(samplesMutex_);
            samples_.clear();
            worst_.clear();
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
            WriteReport(out, configCopy, QueryRunInfo(), Snapshot(), FailedRecordCount(), StalledClockCount(), captured, WorstIterations());
        }

        void WriteTraceTo(std::ostream& out) const { WriteTraceJson(out, WorstIterations()); }

       private:
        Registry() {
            ApplyEnvironmentOverrides(config_);
            PublishHotConfig(config_);
            // Register the fallback report after the process-lifetime instance exists.
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

        // Resolve an explicit budget label first. Otherwise select the sole
        // host-only loop scope and reject ambiguous candidates.
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

        // Keep at most numWorst iterations in descending duration order.
        void RetainIfWorst(std::uint64_t index, double spanMs, std::size_t numWorst) {
            if (worst_.size() >= numWorst && spanMs <= worst_.back().spanMs) return;
            IterationRecord kept;
            kept.index = index;
            kept.spanMs = spanMs;
            kept.spans = iterationSpans_;
            auto position = worst_.begin();
            while (position != worst_.end() && position->spanMs >= spanMs) ++position;
            worst_.insert(position, std::move(kept));
            if (worst_.size() > numWorst) worst_.resize(numWorst);
        }

#if CADENCE_HAS_CUDA
        // Record one anchor per active device and timestamp the midpoint around
        // its synchronization. anchorMutex_ serializes concurrent flushes.
        std::vector<TraceAnchor> RecordTraceAnchors(const std::vector<std::vector<DeviceRecord>>& batches) {
            std::lock_guard<std::mutex> lock(anchorMutex_);
            std::vector<cudaStream_t> streams;
            std::vector<bool> present;
            for (const std::vector<DeviceRecord>& batch : batches) {
                for (const DeviceRecord& record : batch) {
                    if (!record.stop) continue;
                    const std::size_t index = record.device < 0 ? 0 : static_cast<std::size_t>(record.device);
                    if (index >= present.size()) {
                        present.resize(index + 1, false);
                        streams.resize(index + 1, nullptr);
                    }
                    if (!present[index]) {
                        present[index] = true;
                        streams[index] = record.stream;
                    }
                }
            }

            std::vector<TraceAnchor> anchors(present.size());
            if (present.empty()) return anchors;

            // cudaEventCreate binds to the current device, so making an anchor for another one means briefly making it current. In the ordinary single-device case the guard never fires.
            const int previousDevice = CurrentDevice();
            int activeDevice = previousDevice;
            for (std::size_t index = 0; index < present.size(); ++index) {
                if (!present[index]) continue;
                const int device = static_cast<int>(index);
                if (device != activeDevice) {
                    if (cudaSetDevice(device) != cudaSuccess) continue;
                    activeDevice = device;
                }
                cudaEvent_t anchor = AnchorEventFor(index);
                if (!anchor || cudaEventRecord(anchor, streams[index]) != cudaSuccess) continue;
                const std::int64_t before = NowNs();
                if (cudaEventSynchronize(anchor) != cudaSuccess) continue;
                const std::int64_t after = NowNs();
                anchors[index].event = anchor;
                anchors[index].hostNs = before + (after - before) / 2;
                anchors[index].valid = true;
            }
            if (activeDevice != previousDevice) cudaSetDevice(previousDevice);
            return anchors;
        }

        // Owned by the registry and reused every flush, so tracing does not churn the event pool. Created with the matching device current.
        cudaEvent_t AnchorEventFor(std::size_t device) {
            if (device >= anchorEvents_.size()) anchorEvents_.resize(device + 1, nullptr);
            if (!anchorEvents_[device] && cudaEventCreate(&anchorEvents_[device]) != cudaSuccess) {
                anchorEvents_[device] = nullptr;
            }
            return anchorEvents_[device];
        }

        // Called with samplesMutex_ held. A trace viewer wants a small dense row index rather than a pointer, and a process uses a handful of streams, so a scan is the whole implementation. Lane 0 is reserved for the host.
        int LaneFor(cudaStream_t stream) {
            for (std::size_t i = 0; i < lanes_.size(); ++i) {
                if (lanes_[i] == stream) return static_cast<int>(i) + 1;
            }
            lanes_.push_back(stream);
            return static_cast<int>(lanes_.size());
        }
#endif

        // Seed each label reservoir deterministically.
        LabelSamples& SamplesFor(LabelId id) {
            if (id >= samples_.size()) {
                const std::size_t first = samples_.size();
                samples_.resize(id + 1);
                for (std::size_t i = first; i <= id; ++i) {
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
        std::vector<IterationRecord> worst_;  // Slowest first, never longer than numWorstIterations.
        std::vector<IterationSpan> iterationSpans_;  // Reused every flush so the steady state does not allocate.
#if CADENCE_HAS_CUDA
        std::vector<cudaStream_t> lanes_;
        mutable std::mutex anchorMutex_;         // Guards anchorEvents_ only, and is never held alongside another lock.
        std::vector<cudaEvent_t> anchorEvents_;  // One per device, reused by every flush that builds a trace.
#endif

        std::mutex threadsMutex_;
        std::vector<std::shared_ptr<ThreadState>> threads_;

        std::atomic<std::uint64_t> flushGeneration_{1};

#if CADENCE_HAS_CUDA
        mutable std::mutex poolMutex_;
        DeviceEventPools eventPools_;
        std::atomic<std::size_t> capturedScopes_{0};
#endif
    };

    // Keep thread state alive through concurrent flushes and retire it on exit.
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
        const std::int64_t elapsedNs = static_cast<std::int64_t>(elapsedMs * 1e6);
        // Place caller-supplied durations immediately before the current time.
        const std::int64_t endNs = NowNs();
        ThreadState& state = TlsState();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.pendingHost.push_back(HostRecord{handle.id, endNs - elapsedNs, elapsedNs});
    }

    // Select one observation in every sampleEvery for each label.
    CADENCE_ALWAYS_INLINE bool ShouldSample(const LabelHandle& handle) {
        const unsigned every = hotConfig.sampleEvery.load(std::memory_order_relaxed);
        if (CADENCE_LIKELY(every <= 1)) return true;
        if (!handle.observations) return true;
        const std::uint64_t index = handle.observations->fetch_add(1, std::memory_order_relaxed);
        return (index % every) == 0;
    }

#if CADENCE_HAS_CUDA
    // Events recorded during capture become graph nodes and cannot be queried or
    // recycled safely, so captured streams are never instrumented.
    CADENCE_ALWAYS_INLINE bool StreamIsCapturing(cudaStream_t stream) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (CADENCE_UNLIKELY(cudaStreamIsCapturing(stream, &status) != cudaSuccess)) {
            // Global capture can surface as an error on the legacy default stream.
            // Clear the probe error before returning.
            cudaGetLastError();
            return true;
        }
        return status != cudaStreamCaptureStatusNone;
    }

    // Take an event from the thread cache, refilling from the device pool.
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

    // Make one sampling decision per stream chain and flush generation. Sampling
    // individual links would charge omitted gaps to the next retained stage.
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
