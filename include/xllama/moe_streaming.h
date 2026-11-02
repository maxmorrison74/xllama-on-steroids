// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/inference_params.h"

#include <string>

namespace xllama {

/// Parameters for the UWP headless MoE smoke probe. The probe runs an existing
/// GGUF model through the resident llama.cpp backend (`run_inference`), so no
/// submodule edits are required. It validates that a model loads on console
/// and that prefill+decode complete, ahead of the native MoE streamer UWP
/// integration.
struct MoeStreamingParams {
    /// Model filename relative to LocalState\models\ (e.g. "lfm25-350m" or
    /// "qwen3-0.5b-q4_k_m.gguf").
    std::string model_name;

    /// Prompt rendered as-is (no chat template).
    std::string prompt = "The capital of Italy is";

    /// Max tokens to generate.
    int n_predict = 64;

    /// Context size.
    int n_ctx = 512;

    /// Thread count (0 = auto-detect).
    int n_threads = 0;

    /// Greedy decode for deterministic, fast smoke output.
    bool greedy = true;

    /// CPU-only GPU layers (0 = CPU, default).
    int n_gpu_layers = 0;
};

/// Run a synchronous smoke inference against a resident GGUF model on console.
/// Writes `moestream-result.csv` and `moestream-result.csv.done` to LocalState.
/// Returns true on successful load+generate, false otherwise.
bool run_moe_streaming(const MoeStreamingParams& params);

} // namespace xllama
