// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#include "expert_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace xllama::experimental {
namespace {
void bump(std::uint64_t& value) {
    if (value != std::numeric_limits<std::uint64_t>::max())
        ++value;
}
} // namespace

ExpertCache::ExpertCache(std::size_t experts, std::size_t slots) {
    if (experts == 0 || experts > 65536 || slots == 0 || slots > experts)
        throw std::runtime_error("invalid expert cache geometry");
    slots_.resize(slots);
    frequencies_.resize(experts);
    requested_.reserve(slots);
}

std::size_t ExpertCache::find(int expert) const {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].expert == expert)
            return i;
    }
    return slots_.size();
}

std::size_t ExpertCache::slot_for(int expert) const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].expert == expert)
            return i;
    return slots_.size();
}

std::size_t ExpertCache::victim() const {
    std::size_t best = slots_.size();
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const auto& slot = slots_[i];
        if (slot.pinned ||
            std::find(requested_.begin(), requested_.end(), slot.expert) != requested_.end())
            continue;
        if (slot.expert < 0)
            return i;
        if (best == slots_.size() ||
            std::make_tuple(frequencies_[slot.expert], slot.last_use, i) <
                std::make_tuple(frequencies_[slots_[best].expert], slots_[best].last_use, best))
            best = i;
    }
    return best;
}

void ExpertCache::plan(const std::vector<std::int32_t>& experts, ExpertPlan& out) {
    if (active_)
        throw std::runtime_error("expert cache already has an outstanding step");
    if (experts.empty() || experts.size() > frequencies_.size())
        throw std::runtime_error("invalid expert request size");
    requested_.clear();
    for (const int expert : experts) {
        if (expert < 0 || static_cast<std::size_t>(expert) >= frequencies_.size())
            throw std::runtime_error("expert id out of range");
        if (std::find(requested_.begin(), requested_.end(), expert) == requested_.end()) {
            if (requested_.size() == slots_.size())
                throw std::runtime_error("routing step needs more experts than cache slots");
            requested_.push_back(expert);
        }
    }
    // All allocations happen before changing slots/counters. Invalid requests
    // leave cache residency and telemetry untouched; no in-flight I/O exists.
    out.bindings.reserve(requested_.size());
    out.slot_ids.reserve(experts.size());
    out.bindings.clear();
    out.slot_ids.clear();
    for (const int expert : requested_) {
        std::size_t slot = find(expert);
        const bool miss = slot == slots_.size();
        if (miss) {
            slot = victim();
            if (slot == slots_.size())
                throw std::runtime_error("expert cache victim invariant failed");
            if (slots_[slot].expert >= 0)
                bump(stats_.evictions);
            slots_[slot] = {expert, false, true, 0};
            bump(stats_.misses);
        } else {
            if (!slots_[slot].valid)
                throw std::runtime_error("expert cache contains unfinished data");
            slots_[slot].pinned = true;
            bump(stats_.hits);
        }
        bump(frequencies_[expert]);
        bump(clock_);
        slots_[slot].last_use = clock_;
        out.bindings.push_back({expert, slot, miss});
    }
    for (const int expert : experts)
        out.slot_ids.push_back(static_cast<std::int32_t>(find(expert)));
    active_ = true;
    bump(stats_.decisions);
}

void ExpertCache::mark_loaded(std::size_t slot) {
    if (!active_ || slot >= slots_.size() || !slots_[slot].pinned || slots_[slot].valid)
        throw std::runtime_error("invalid expert load completion");
    slots_[slot].valid = true;
    bump(stats_.completed_loads);
}

void ExpertCache::finish() {
    if (!active_)
        throw std::runtime_error("no expert step to finish");
    for (const auto& slot : slots_) {
        if (slot.pinned && !slot.valid)
            throw std::runtime_error("cannot release an unfinished expert read");
    }
    for (auto& slot : slots_)
        slot.pinned = false;
    active_ = false;
}

} // namespace xllama::experimental
