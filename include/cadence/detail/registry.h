// cadence — the deferred-flush registry.
//
// The hot path does as little as possible: record two events (or read a
// steady_clock), then push a small POD under a mutex. Nothing synchronizes,
// nothing allocates a string, nothing queries the device. All of the real work
// -- synchronizing events, computing elapsed times, discarding warmup,
// aggregating statistics -- happens in Flush(), which the application calls at
// a point where a synchronization is already acceptable.
//
// A profiler that serializes the pipeline is measuring itself.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "cadence/detail/config.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/report.h"
#include "cadence/detail/stats.h"

#if CADENCE_HAS_CUDA
#include "cadence/detail/event_pool.h"
#endif

namespace cadence {
namespace detail {

// Labels are stored as raw pointers and only copied into a std::string during Flush(). The documented contract is that a label outlives the Flush() that consumes it -- string literals, which is what the macros pass, always do.
struct HostRecord {
  const char* label;
  double elapsedMs;
};

#if CADENCE_HAS_CUDA
struct DeviceRecord {
  const char* label;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  // CPU time spent inside the scope, for launch-bound vs compute-bound comparison against elapsedMs.
  double hostIssueMs;
};
#endif

// Accumulated samples for one label, across every flush so far.
struct LabelSamples {
  std::vector<double> deviceMs;
  std::vector<double> hostMs;   // Host scopes, or CPU-issue for device scopes.
  std::uint64_t seen = 0;       // Total observations, warmup included.
  std::uint64_t discarded = 0;  // Observations dropped as warmup.
  bool hasDevice = false;
  bool hasHostIssue = false;
};

class Registry {
 public:
  static Registry& Instance() {
    static Registry instance;
    return instance;
  }

  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  const Config& GetConfig() const { return config_; }

  void Configure(const Config& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    ApplyEnvironmentOverrides(config_);
  }

  bool IsEnabled() const { return config_.enabled; }

  void RecordHost(const char* label, double elapsedMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingHost_.push_back(HostRecord{label, elapsedMs});
  }

#if CADENCE_HAS_CUDA
  // Callers acquire events through the registry so the pool stays behind one lock. Returns nullptr when the runtime is out of events.
  cudaEvent_t AcquireEvent() {
    std::lock_guard<std::mutex> lock(mutex_);
    return eventPool_.Acquire();
  }

  void RecordDevice(const DeviceRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingDevice_.push_back(record);
  }
#endif

  // Consumes every pending record. This is a synchronization point: it waits for the recorded stop events, which is why it belongs at a frame or loop boundary and never inside the measured region.
  void Flush() {
    std::vector<HostRecord> hostBatch;
#if CADENCE_HAS_CUDA
    std::vector<DeviceRecord> deviceBatch;
#endif
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hostBatch.swap(pendingHost_);
#if CADENCE_HAS_CUDA
      deviceBatch.swap(pendingDevice_);
#endif
    }

#if CADENCE_HAS_CUDA
    // Two passes. First wait for every stop event, so the elapsed-time queries in the second pass never block. Events on one stream complete in record order, so this is usually a single real wait followed by no-ops.
    for (const DeviceRecord& record : deviceBatch) {
      if (record.stop) cudaEventSynchronize(record.stop);
    }
#endif

    {
      std::lock_guard<std::mutex> lock(mutex_);
#if CADENCE_HAS_CUDA
      for (const DeviceRecord& record : deviceBatch) {
        float elapsedMs = 0.0f;
        const bool valid =
            record.start && record.stop &&
            cudaEventElapsedTime(&elapsedMs, record.start, record.stop) == cudaSuccess;
        if (valid) {
          LabelSamples& samples = samples_[record.label];
          samples.hasDevice = true;
          samples.hasHostIssue = true;
          if (KeepSample(samples)) {
            samples.deviceMs.push_back(static_cast<double>(elapsedMs));
            samples.hostMs.push_back(record.hostIssueMs);
          }
        } else if (record.start || record.stop) {
          ++failedRecords_;
        }
        eventPool_.Release(record.start);
        eventPool_.Release(record.stop);
      }
#endif
      for (const HostRecord& record : hostBatch) {
        LabelSamples& samples = samples_[record.label];
        if (KeepSample(samples)) samples.hostMs.push_back(record.elapsedMs);
      }
    }
  }

  // Flush() first if you want the current loop's records included.
  std::vector<Stats> Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Stats> results;
    results.reserve(samples_.size());
    for (const auto& entry : samples_) {
      const LabelSamples& samples = entry.second;
      if (samples.hasDevice) {
        results.push_back(
            ComputeStats(entry.first, ScopeKind::Device, samples.deviceMs, samples.discarded));
      }
      if (!samples.hostMs.empty()) {
        // For a device scope this row is the CPU-issue side of the same label: compare it against the device row to see launch-bound vs compute-bound.
        results.push_back(
            ComputeStats(entry.first, ScopeKind::Host, samples.hostMs, samples.discarded));
      }
    }
    return results;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    failedRecords_ = 0;
  }

  std::size_t FailedRecordCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failedRecords_;
  }

  // Suppresses the exit-time write: the application reported explicitly, and that report was taken while the CUDA runtime was certainly still alive.
  void MarkReported() {
    std::lock_guard<std::mutex> lock(mutex_);
    reported_ = true;
  }

  void WriteTo(std::ostream& out) const {
    Config configCopy;
    std::size_t failed = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      configCopy = config_;
      failed = failedRecords_;
    }
    WriteReportHeader(out, configCopy, QueryRunInfo(), failed);
    WriteStatsCsv(out, Snapshot());
  }

 private:
  Registry() {
    ApplyEnvironmentOverrides(config_);
    // Last-resort output for applications that never call Report(). It runs before this object is destroyed, because atexit handlers and static destructors unwind in one interleaved reverse-order sequence and this registration happens after construction.
    std::atexit(&Registry::AtExitHandler);
  }

  static void AtExitHandler() {
    Registry& registry = Instance();
    const Config config = registry.GetConfig();
    if (!config.writeOnExit || config.outputPath.empty()) return;
    {
      std::lock_guard<std::mutex> lock(registry.mutex_);
      if (registry.reported_) return;
    }
    // The CUDA runtime may already be shutting down; Flush() tolerates that and the report simply loses whatever was still pending.
    registry.Flush();
    std::ofstream out(config.outputPath);
    if (out) registry.WriteTo(out);
  }

  // Warmup discard. Called with mutex_ held; increments the per-label counter and reports whether this observation should contribute to statistics.
  bool KeepSample(LabelSamples& samples) {
    const std::uint64_t index = samples.seen++;
    if (index < config_.warmupIterations) {
      samples.discarded = index + 1;
      return false;
    }
    return true;
  }

  mutable std::mutex mutex_;
  Config config_;
  std::vector<HostRecord> pendingHost_;
  std::map<std::string, LabelSamples> samples_;
  std::size_t failedRecords_ = 0;
  bool reported_ = false;
#if CADENCE_HAS_CUDA
  std::vector<DeviceRecord> pendingDevice_;
  EventPool eventPool_;
#endif
};

}  // namespace detail
}  // namespace cadence
