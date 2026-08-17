// Label interning. Scope macros cache a compact handle at each call site;
// directly constructed scopes resolve their labels on every execution.
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
            // Process-lifetime storage avoids static destruction order hazards.
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

// Resolve a label once per call site using thread-safe static initialization.
#define CADENCE_DETAIL_LABEL(label)                                  \
    ([]() -> const ::cadence::detail::LabelHandle& {                 \
        static const ::cadence::detail::LabelHandle handle =         \
            ::cadence::detail::LabelTable::Instance().Intern(label); \
        return handle;                                               \
    }())
