// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#include "moe_streaming.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t positive(const char* text) {
    const std::string value = text;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("budgets and slots must be positive integers");
    const auto number = std::stoull(value);
    if (number == 0 || number > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("argument out of range");
    return static_cast<std::size_t>(number);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(
            stderr, "Experimental HOST qwen3moe streaming; NOT a UWP build or speed claim.\n"
                    "Usage: xllama-moe-streaming-spike MODEL.gguf WEIGHT_BUDGET_MiB SLOTS PROMPT\n"
                    "Budget excludes KV/compute/runtime. Physical ubatch=1; raw prompt; greedy.\n");
        return 2;
    }
    llama_backend_init();
    int result = 0;
    try {
        const auto mib = positive(argv[2]);
        const auto slots = positive(argv[3]);
        if (mib > std::numeric_limits<std::size_t>::max() / (1024 * 1024))
            throw std::runtime_error("weight budget overflow");
        xllama::experimental::MoeStreamingModel model(argv[1], mib * 1024 * 1024, slots);
        const auto* vocab = llama_model_get_vocab(model.model());
        const std::string prompt = argv[4];
        if (prompt.size() > 4096)
            throw std::runtime_error("spike prompt is limited to 4096 bytes");
        std::vector<llama_token> tokens(128);
        const int count =
            llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(),
                           static_cast<int>(tokens.size()), true, true);
        if (count <= 0)
            throw std::runtime_error("prompt must tokenize to 1..128 tokens");
        tokens.resize(count);
        model.create_context(256, 32, 2);
        for (std::size_t start = 0; start < tokens.size(); start += 32)
            model.decode(llama_batch_get_one(
                tokens.data() + start,
                static_cast<int>(std::min<std::size_t>(32, tokens.size() - start))));
        for (int step = 0; step < 16; ++step) {
            const auto* logits = model.logits();
            const auto size = llama_vocab_n_tokens(vocab);
            if (!std::all_of(logits, logits + size,
                             [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("non-finite MoE logits");
            auto next = static_cast<llama_token>(std::max_element(logits, logits + size) - logits);
            if (llama_vocab_is_eog(vocab, next))
                break;
            std::vector<char> piece(64);
            int length = llama_token_to_piece(vocab, next, piece.data(),
                                              static_cast<int>(piece.size()), 0, true);
            if (length < 0) {
                piece.resize(-length);
                length = llama_token_to_piece(vocab, next, piece.data(),
                                              static_cast<int>(piece.size()), 0, true);
            }
            if (length < 0)
                throw std::runtime_error("cannot detokenize MoE output");
            std::fwrite(piece.data(), 1, length, stdout);
            std::fflush(stdout);
            if (step + 1 < 16)
                model.decode(llama_batch_get_one(&next, 1));
        }
        const auto& stats = model.stats();
        std::fprintf(stderr,
                     "\nMoE allocated weights: common=%zu cache=%zu logical=%zu; "
                     "expert_read_bytes=%llu decisions=%llu hits=%llu misses=%llu evictions=%llu\n",
                     stats.common_bytes, stats.expert_cache_bytes, stats.logical_weight_bytes,
                     static_cast<unsigned long long>(stats.expert_read_bytes),
                     static_cast<unsigned long long>(stats.cache.decisions),
                     static_cast<unsigned long long>(stats.cache.hits),
                     static_cast<unsigned long long>(stats.cache.misses),
                     static_cast<unsigned long long>(stats.cache.evictions));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE streaming spike failed: %s\n", error.what());
        result = 1;
    }
    llama_backend_free();
    return result;
}
