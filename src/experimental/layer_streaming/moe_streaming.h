// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "expert_cache.h"
#include "llama.h"
#include "moe_file.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xllama::experimental {

struct MoeStreamingStats {
    std::size_t common_bytes = 0;
    std::size_t expert_cache_bytes = 0;
    std::size_t logical_weight_bytes = 0;
    std::uint64_t common_read_bytes = 0;
    std::uint64_t expert_read_bytes = 0;
    bool file_unbuffered = false;
    std::uint64_t chunk_visits = 0;
    std::uint64_t synchronized_checkpoints = 0;
    ExpertCacheStats cache;
};

// Pinned-internals HOST correctness prototype, not the UWP implementation.
// Dense common tensors stay resident; routed FFNs use persistent per-layer
// expert slots. CPU only. Weight budget excludes metadata, KV, compute and UI.
// A whole routing step is protected before any victim is chosen; an expert
// slot becomes valid only after every gate/up/down projection for it has
// been read. Router expert counts, weights and probabilities are unchanged:
// selected expert IDs are temporarily translated to cache slots for the
// duration of one CPU MUL_MAT_ID node, then restored.
class MoeStreamingModel {
  public:
    MoeStreamingModel(const std::string& path, std::size_t weight_budget_bytes,
                      std::size_t slots_per_layer, std::size_t physical_ubatch = 1);
    ~MoeStreamingModel();
    MoeStreamingModel(const MoeStreamingModel&) = delete;
    MoeStreamingModel& operator=(const MoeStreamingModel&) = delete;

    void create_context(std::uint32_t n_ctx, std::uint32_t batch_size, int threads);
    void decode(llama_batch batch);
    void cancel() noexcept;
    const float* logits() const;
    const llama_model* model() const;
    const MoeStreamingStats& stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xllama::experimental
