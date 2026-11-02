// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllama::experimental {

struct ExpertBinding {
    int expert = -1;
    std::size_t slot = 0;
    bool needs_read = false;
};

struct ExpertPlan {
    // One binding per distinct expert; slot_ids preserves the router's order,
    // including duplicates. Only slot_ids, never router probabilities, changes.
    std::vector<ExpertBinding> bindings;
    std::vector<std::int32_t> slot_ids;
};

struct ExpertCacheStats {
    std::uint64_t decisions = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t completed_loads = 0;
    std::uint64_t evictions = 0;
};

// Bookkeeping only, no I/O or tensors. Original implementation of the cache
// strategy reviewed in docs/ssd-inference-assessment.md §9. A whole requested
// set is protected BEFORE selecting victims. Frequencies survive eviction.
// Single owner, one outstanding step. No slot may be reused while a step is live.
class ExpertCache {
  public:
    ExpertCache(std::size_t experts, std::size_t slots);
    void plan(const std::vector<std::int32_t>& experts, ExpertPlan& out);
    void mark_loaded(std::size_t slot);
    void finish();
    bool active() const noexcept {
        return active_;
    }
    const ExpertCacheStats& stats() const noexcept {
        return stats_;
    }

    // Slot holding `expert`, or slots_.size() if it is not resident.
    std::size_t slot_for(int expert) const;

  private:
    struct Slot {
        int expert = -1;
        bool valid = false;
        bool pinned = false;
        std::uint64_t last_use = 0;
    };
    std::size_t find(int expert) const;
    std::size_t victim() const;
    std::vector<Slot> slots_;
    std::vector<std::uint64_t> frequencies_;
    std::vector<std::int32_t> requested_;
    std::uint64_t clock_ = 0;
    bool active_ = false;
    ExpertCacheStats stats_;
};

} // namespace xllama::experimental
