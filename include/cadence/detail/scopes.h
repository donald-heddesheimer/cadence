// cadence: the RAII timers.
//
// Three of them, and the difference between them is how many CUDA events they record, because that is what the instrumentation costs.
//   ScopedHost    no events. A steady_clock bracket around a CPU span.
//   ScopedKernel  two events. Measures exactly the work inside the scope.
//   ScopedStage   one event. Measures from the end of the previous stage.
//
// None of them synchronize. Every one of them appends to a thread-local buffer and returns.
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

#include "cadence/detail/labels.h"
#include "cadence/detail/nvtx.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/registry.h"

namespace cadence {
    namespace detail {

    CADENCE_ALWAYS_INLINE std::int64_t ElapsedNs(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count();
    }

    }  // namespace detail

    // A host-side span measured with steady_clock. Usable in plain C++ translation units with no CUDA in sight.
    class ScopedHost {
       public:
        explicit ScopedHost(const detail::LabelHandle& label)
            : labelId_(label.id),
              active_(detail::hotConfig.enabled.load(std::memory_order_relaxed) && detail::ShouldSample(label)),
              nvtx_(label.name, active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
            if (CADENCE_LIKELY(active_)) start_ = std::chrono::steady_clock::now();
        }

        // Interns on every construction, and takes a lock to do it. CADENCE_SCOPE resolves the label once per call site instead; prefer it.
        explicit ScopedHost(const char* label)
            : ScopedHost(detail::LabelTable::Instance().Intern(label)) {}

        ~ScopedHost() {
            if (!active_) return;
            const auto end = std::chrono::steady_clock::now();
            const std::int64_t elapsedNs = detail::ElapsedNs(start_, end);
            detail::ThreadState& state = detail::TlsState();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pendingHost.push_back(detail::HostRecord{labelId_, elapsedNs});
        }

        ScopedHost(const ScopedHost&) = delete;
        ScopedHost& operator=(const ScopedHost&) = delete;

       private:
        detail::LabelId labelId_;
        bool active_;
        detail::NvtxRange nvtx_;
        std::chrono::steady_clock::time_point start_{};
    };

#if CADENCE_HAS_CUDA

    // Times GPU work with paired CUDA events. 
    class ScopedKernel {
       public:
        explicit ScopedKernel(const detail::LabelHandle& label, cudaStream_t stream = 0)
            : labelId_(label.id),
              stream_(stream),
              active_(detail::hotConfig.enabled.load(std::memory_order_relaxed) &&
                      detail::ShouldSample(label)),
              nvtx_(label.name,
                    active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
            if (!active_) return;
            // Nothing may be recorded onto a stream that is being captured into a graph, so the scope stands down and says so in the report rather than corrupting the capture.
            if (CADENCE_UNLIKELY(detail::StreamIsCapturing(stream_))) {
                active_ = false;
                detail::Registry::Instance().NoteCapturedScope();
                return;
            }
            device_ = detail::CurrentDevice();
            detail::ThreadState& state = detail::TlsState();
            start_ = detail::TakeEvent(state, device_);
            stop_ = detail::TakeEvent(state, device_);
            if (CADENCE_UNLIKELY(!start_ || !stop_)) {
                // Out of events: stay quiet rather than taking the application down, but hand back whatever was taken so the count stays honest.
                active_ = false;
                Append(state, detail::DeviceRecord{labelId_, true, device_, stream_, start_, stop_, 0});
                start_ = stop_ = nullptr;
                return;
            }
            hostStart_ = std::chrono::steady_clock::now();
            cudaEventRecord(start_, stream_);
        }

        explicit ScopedKernel(const char* label, cudaStream_t stream = 0) : ScopedKernel(detail::LabelTable::Instance().Intern(label), stream) {}

        ~ScopedKernel() {
            if (!active_) return;
            // Capture can open inside the scope as easily as before it, and a stop event recorded into a graph is exactly as damaging as a start one. Both events go back to the thread's own cache unrecorded; the start event is already spent, which costs this one observation and nothing else. Checked before anything else here so that only this one call sits between the work and the stop event that closes it.
            if (CADENCE_UNLIKELY(detail::StreamIsCapturing(stream_))) {
                detail::ThreadState& state = detail::TlsState();
                detail::GiveBackEvent(state, device_, start_);
                detail::GiveBackEvent(state, device_, stop_);
                detail::Registry::Instance().NoteCapturedScope();
                return;
            }
            cudaEventRecord(stop_, stream_);
            const auto hostEnd = std::chrono::steady_clock::now();
            detail::ThreadState& state = detail::TlsState();
            Append(state, detail::DeviceRecord{labelId_, true, device_, stream_, start_, stop_, detail::ElapsedNs(hostStart_, hostEnd)});
        }

        ScopedKernel(const ScopedKernel&) = delete;
        ScopedKernel& operator=(const ScopedKernel&) = delete;

       private:
        static void Append(detail::ThreadState& state, const detail::DeviceRecord& record) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pendingDevice.push_back(record);
        }

        detail::LabelId labelId_;
        cudaStream_t stream_;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
        int device_ = 0;
        bool active_;
        detail::NvtxRange nvtx_;
        std::chrono::steady_clock::time_point hostStart_{};
    };

    // Times one stage of a chain, using the previous stage's stop event as its start. 
    class ScopedStage {
       public:
        explicit ScopedStage(const detail::LabelHandle& label, cudaStream_t stream = 0)
            : labelId_(label.id),
              stream_(stream),
              active_(detail::hotConfig.enabled.load(std::memory_order_relaxed) &&
                      detail::ShouldSampleChain(label, stream)),
              nvtx_(label.name,
                    active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
            if (!active_) return;
            if (CADENCE_UNLIKELY(detail::StreamIsCapturing(stream_))) {
                active_ = false;
                detail::Registry::Instance().NoteCapturedScope();
                return;
            }
            device_ = detail::CurrentDevice();
            detail::ThreadState& state = detail::TlsState();
            detail::StreamChain* chain = state.FindChain(stream_);
            generation_ = detail::Registry::Instance().FlushGeneration();
            // A tail from an earlier flush is a handle to an event already back in the pool, so age is checked before the pointer is trusted. A tail from another device cannot be paired with a stop event from this one, so it is treated the same way: the chain restarts here.
            if (chain->generation != generation_ || !chain->tail || chain->device != device_) {
                cudaEvent_t head = detail::TakeEvent(state, device_);
                if (CADENCE_UNLIKELY(!head)) {
                    active_ = false;
                    chain->tail = nullptr;
                    return;
                }
                cudaEventRecord(head, stream_);
                chain->tail = head;
                chain->generation = generation_;
                chain->device = device_;
                ownsStart_ = true;
            }
            start_ = chain->tail;
            hostStart_ = std::chrono::steady_clock::now();
        }

        explicit ScopedStage(const char* label, cudaStream_t stream = 0)
            : ScopedStage(detail::LabelTable::Instance().Intern(label), stream) {}

        ~ScopedStage() {
            if (!active_) return;
            detail::ThreadState& state = detail::TlsState();
            // Capture opened inside the scope. Break the chain and carry any head this stage owns back to the pool; recording the stop would bake both events into the graph.
            if (CADENCE_UNLIKELY(detail::StreamIsCapturing(stream_))) {
                state.FindChain(stream_)->tail = nullptr;
                if (ownsStart_) detail::GiveBackEvent(state, device_, start_);
                detail::Registry::Instance().NoteCapturedScope();
                return;
            }
            cudaEvent_t stop = detail::TakeEvent(state, device_);
            const auto hostEnd = std::chrono::steady_clock::now();
            detail::StreamChain* chain = state.FindChain(stream_);
            if (CADENCE_UNLIKELY(!stop)) {
                // Break the chain rather than leave a head nobody will ever release; the record carries the orphan back to the pool via Flush().
                chain->tail = nullptr;
                if (ownsStart_) {
                    Append(state, detail::DeviceRecord{labelId_, true, device_, stream_, start_, nullptr, 0});
                }
                return;
            }
            cudaEventRecord(stop, stream_);
            chain->tail = stop;
            chain->generation = generation_;
            chain->device = device_;
            Append(state, detail::DeviceRecord{labelId_, ownsStart_, device_, stream_, start_, stop, detail::ElapsedNs(hostStart_, hostEnd)});
        }

        ScopedStage(const ScopedStage&) = delete;
        ScopedStage& operator=(const ScopedStage&) = delete;

       private:
        static void Append(detail::ThreadState& state, const detail::DeviceRecord& record) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pendingDevice.push_back(record);
        }

        detail::LabelId labelId_;
        cudaStream_t stream_;
        cudaEvent_t start_ = nullptr;
        int device_ = 0;
        bool active_;
        // True only for the stage that opened the chain: every other stage borrows its predecessor's stop event, and the pool must take each event back exactly once.
        bool ownsStart_ = false;
        std::uint64_t generation_ = 0;
        detail::NvtxRange nvtx_;
        std::chrono::steady_clock::time_point hostStart_{};
    };

#endif  // CADENCE_HAS_CUDA

}  // namespace cadence
