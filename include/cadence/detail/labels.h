// cadence: label interning.
//
// A label is resolved to a small integer once per call site, not once per execution: the scope macros hold the result in a function-local static, so the second and every later pass through a scope costs one guard-variable load. Three things follow from that.
//
//   1. A record carries a 4-byte id instead of an 8-byte pointer.
//   2. Flush() indexes a vector instead of hashing a string into a map.
//   3. The label no longer has to outlive the flush that consumes it, because
//      the string was copied when it was interned.
//
// Constructing a scope class directly, rather than through the macros, interns on every construction and takes a lock to do it. That is the slow path and the reason the macros exist.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cadence {
    namespace detail {
    using LabelId = std::uint32_t;

    inline constexpr LabelId INVALID_LABEL_ID = 0xFFFFFFFFu;

    // What a call site caches. 
    struct LabelHandle {
        LabelId id = INVALID_LABEL_ID;
        std::atomic<std::uint64_t>* observations = nullptr;
        const char* name = "";
    };

    class LabelTable {
       public:
        static LabelTable& Instance() {
            // labels are referenced by the exit-time report and by thread-local state whose destruction order against a function-local static is not something worth betting a crash on.
            static LabelTable* instance = new LabelTable();
            return *instance;
        }

        LabelTable(const LabelTable&) = delete;
        LabelTable& operator=(const LabelTable&) = delete;

        LabelHandle Intern(const char* label) {
            const std::string text(label ? label : "");
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = ids_.find(text);
            if (found != ids_.end()) return Handle(found->second);
            const LabelId id = static_cast<LabelId>(names_.size());
            names_.push_back(text);
            observations_.emplace_back(0);
            ids_.emplace(text, id);
            return Handle(id);
        }

        std::size_t Count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return names_.size();
        }

        // Copied out under the lock. Only Snapshot() needs names, and it is not on any hot path.
        std::vector<std::string> Names() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return std::vector<std::string>(names_.begin(), names_.end());
        }

        std::uint64_t Observations(LabelId id) const {
            std::lock_guard<std::mutex> lock(mutex_);
            if (id >= observations_.size()) return 0;
            return observations_[id].load(std::memory_order_relaxed);
        }

       private:
        LabelTable() = default;

        // Called with mutex_ held.
        LabelHandle Handle(LabelId id) { return LabelHandle{id, &observations_[id], names_[id].c_str()}; }

        mutable std::mutex mutex_;
        std::deque<std::string> names_;
        std::deque<std::atomic<std::uint64_t>> observations_;
        std::unordered_map<std::string, LabelId> ids_;
    };

    }  // namespace detail
}  // namespace cadence

// Resolves a label to a handle once per call site. The static is function-local so it is initialized on first pass and thread-safe by C++11 rules; every later pass is a load and a predictable branch.
#define CADENCE_DETAIL_LABEL(label)                                  \
    ([]() -> const ::cadence::detail::LabelHandle& {                 \
        static const ::cadence::detail::LabelHandle handle =         \
            ::cadence::detail::LabelTable::Instance().Intern(label); \
        return handle;                                               \
    }())
