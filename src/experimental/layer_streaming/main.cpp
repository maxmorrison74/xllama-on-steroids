// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "layer_streaming.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Experimental host CPU streaming; NOT an Xbox build.\n"
                             "Usage: xllama-streaming-spike MODEL.gguf WEIGHT_BUDGET_MiB PROMPT\n"
                             "Weight budget excludes KV, compute scratch and runtime overhead.\n");
        return 2;
    }
    llama_backend_init();
    int result = 0;
    try {
        const std::string text_budget = argv[2];
        if (text_budget.empty() || text_budget.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("weight budget must be a positive integer MiB count");
        const auto mib = std::stoull(text_budget);
        if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1024 * 1024))
            throw std::runtime_error("invalid weight budget");
        xllama::experimental::LayerStreamingModel model(argv[1], mib * 1024 * 1024);
        const auto* vocab = llama_model_get_vocab(model.model());
        const std::string prompt = argv[3];
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
        for (std::size_t start = 0; start < tokens.size(); start += 32) {
            const auto size = std::min<std::size_t>(32, tokens.size() - start);
            model.decode(llama_batch_get_one(tokens.data() + start, static_cast<int>(size)));
        }
        for (int step = 0; step < 16; ++step) {
            const float* logits = model.logits();
            if (!std::all_of(logits, logits + llama_vocab_n_tokens(vocab),
                             [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("streaming model produced non-finite logits");
            const auto next = static_cast<llama_token>(
                std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits);
            if (llama_vocab_is_eog(vocab, next))
                break;
            std::vector<char> piece(64);
            int size = llama_token_to_piece(vocab, next, piece.data(),
                                            static_cast<int>(piece.size()), 0, true);
            if (size < 0) {
                piece.resize(-size);
                size = llama_token_to_piece(vocab, next, piece.data(),
                                            static_cast<int>(piece.size()), 0, true);
            }
            if (size < 0)
                throw std::runtime_error("cannot detokenize generated token");
            std::fwrite(piece.data(), 1, size, stdout);
            std::fflush(stdout);
            auto token = next;
            if (step + 1 < 16)
                model.decode(llama_batch_get_one(&token, 1));
        }
        const auto& stats = model.stats();
        std::fprintf(stderr,
                     "\nStreaming weights: resident=%zu, each_slot=%zu, "
                     "logical_total=%zu bytes; reads=%llu bytes, layer_loads=%llu\n",
                     stats.resident_bytes, stats.slot_bytes, stats.total_weight_bytes,
                     static_cast<unsigned long long>(stats.bytes_read),
                     static_cast<unsigned long long>(stats.layer_loads));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "streaming spike failed: %s\n", error.what());
        result = 1;
    }
    llama_backend_free();
    return result;
}
