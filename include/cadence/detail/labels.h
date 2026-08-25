// Label interning. Scope macros cache a compact handle at each call site;
// directly constructed scopes resolve their labels on every execution.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cadence/detail/config.h"

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
            // A label already in the table always resolves; only a new one can be refused, so a runaway label built at runtime cannot displace the ones the loop actually uses.
            const std::size_t cap = hotConfig.maxLabels.load(std::memory_order_relaxed);
            if (cap != 0 && names_.size() >= cap) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return LabelHandle{};
            }
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

        // Intern calls refused because the table was already at Config::maxLabels. A macro call site interns once, so it contributes at most one; a scope constructed with a runtime label interns on every execution and contributes one each time.
        std::size_t DroppedCount() const { return dropped_.load(std::memory_order_relaxed); }

       private:
        LabelTable() = default;

        // Called with mutex_ held.
        LabelHandle Handle(LabelId id) { return LabelHandle{id, &observations_[id], names_[id].c_str()}; }

        mutable std::mutex mutex_;
        std::deque<std::string> names_;
        std::deque<std::atomic<std::uint64_t>> observations_;
        std::unordered_map<std::string, LabelId> ids_;
        // Atomic so the report can read it without taking the label lock.
        std::atomic<std::size_t> dropped_{0};
    };

    // Rejects a label the macros cannot cache safely.
    //
    // The contract is that the characters do not change and outlive the flush, so the test is a const character array: a string literal, or any other const array. A mutable array fails, because nothing stops its contents being rewritten after the handle is cached. A pointer fails because its target is whatever it happened to point at the first time this call site ran. Keying on array-ness alone would let `char label[32]` through.
    template <typename T>
    constexpr const char* RequireStableLabel(T&& label) {
        using Label = std::remove_reference_t<T>;
        static_assert(std::is_array<Label>::value && std::is_const<std::remove_extent_t<Label>>::value,
                      "cadence: a macro label must be a string literal, or another const character array. The macro "
                      "interns it once per call site, so a label whose contents can change would file every scope "
                      "under its first value. Use cadence::ScopedHost or cadence::ScopedKernel directly for a label "
                      "computed at runtime.");
        return label;
    }

    // True for a handle the table refused. Such a scope records nothing: there is no row for it to land in.
    inline bool LabelDropped(const LabelHandle& handle) { return handle.id == INVALID_LABEL_ID; }

    }  // namespace detail
}  // namespace cadence

// Resolve a label once per call site using thread-safe static initialization.
#define CADENCE_DETAIL_LABEL(label)                             \
    ([]() -> const ::cadence::detail::LabelHandle& {            \
        static const ::cadence::detail::LabelHandle handle =    \
            ::cadence::detail::LabelTable::Instance().Intern(   \
                ::cadence::detail::RequireStableLabel(label));  \
        return handle;                                          \
    }())
