// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xllama::experimental {

struct StreamingStats {
    std::size_t resident_bytes = 0;
    std::size_t slot_bytes = 0;
    std::size_t total_weight_bytes = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t layer_loads = 0;
};

// Research adapter tied to the pinned llama.cpp internals. CPU only, no repack,
// no MTP/LoRA, no concurrent contexts. Not a shipping Session implementation.
// Own the context inside this object so no callback or weight buffer outlives it.
class LayerStreamingModel {
  public:
    LayerStreamingModel(const std::string& path, std::size_t weight_budget_bytes);
    ~LayerStreamingModel();
    LayerStreamingModel(const LayerStreamingModel&) = delete;
    LayerStreamingModel& operator=(const LayerStreamingModel&) = delete;

    void create_context(std::uint32_t n_ctx, std::uint32_t batch_size, int threads);
    void decode(llama_batch batch);
    // The only operation allowed concurrently with decode; cancellation is sticky.
    void cancel() noexcept;
    const float* logits() const;
    const llama_model* model() const;
    const StreamingStats& stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xllama::experimental
