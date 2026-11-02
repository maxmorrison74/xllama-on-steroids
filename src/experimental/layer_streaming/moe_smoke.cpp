// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
// Synthetic random weights: parity/cache tests, never model quality evidence.
#include "moe_streaming.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace xllama::experimental;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function> void rejects(Function&& action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

ExpertPlan step(ExpertCache& cache, const std::vector<std::int32_t>& ids) {
    ExpertPlan plan;
    cache.plan(ids, plan);
    for (const auto& binding : plan.bindings) {
        if (binding.needs_read)
            cache.mark_loaded(binding.slot);
    }
    cache.finish();
    return plan;
}

void cache_tests() {
    rejects([] { ExpertCache invalid(0, 1); }, "empty expert space accepted");
    rejects([] { ExpertCache invalid(8, 9); }, "too many slots accepted");
    ExpertCache cache(8, 2);
    auto first = step(cache, {0, 1});
    require(first.slot_ids == std::vector<std::int32_t>({0, 1}), "initial slot assignment");
    step(cache, {0});
    step(cache, {0});
    // Expert 1 is cold, but this WHOLE request protects it before the miss 2.
    auto protect = step(cache, {2, 1});
    require(protect.slot_ids[1] == 1 && protect.bindings[1].needs_read == false,
            "a later requested hit was evicted by an earlier miss");
    const auto before = cache.stats();
    ExpertPlan bad;
    rejects([&] { cache.plan({0, 1, 2}, bad); }, "too few slots accepted");
    rejects([&] { cache.plan({8}, bad); }, "out-of-range expert accepted");
    require(cache.stats().decisions == before.decisions && cache.stats().misses == before.misses,
            "invalid routing changed cache telemetry");
    ExpertCache duplicate(8, 2);
    auto duplicates = step(duplicate, {3, 3, 4});
    require(duplicates.bindings.size() == 2 && duplicates.slot_ids[0] == duplicates.slot_ids[1] &&
                duplicate.stats().misses == 2,
            "duplicate IDs changed request order or counters");
    ExpertCache pending(8, 2);
    ExpertPlan plan;
    pending.plan({3, 4}, plan);
    rejects([&] { pending.finish(); }, "unfinished reads were released");
    rejects([&] { pending.plan({5}, bad); }, "in-flight slots were reassigned");
    for (const auto& binding : plan.bindings)
        pending.mark_loaded(binding.slot);
    pending.finish();
    require(step(pending, {3, 4}).bindings[0].needs_read == false, "ready expert did not hit");

    // Ghost history must survive eviction; a hot readmitted expert is not reset.
    ExpertCache ghost(8, 3);
    for (int i = 0; i < 5; ++i)
        step(ghost, {0});
    for (int i = 0; i < 3; ++i)
        step(ghost, {1});
    for (int i = 0; i < 3; ++i)
        step(ghost, {2});
    const auto expert1_slot = step(ghost, {1}).slot_ids[0];
    step(ghost, {2});
    // Frequencies 0=5, 1=4, 2=4, with 1 least recently used among the cold pair.
    step(ghost, {3});
    const auto readmitted = step(ghost, {1});
    require(readmitted.bindings[0].needs_read, "ghost-history test did not evict expert 1");
    step(ghost, {4});
    const auto preserved = step(ghost, {1});
    require(!preserved.bindings[0].needs_read && preserved.slot_ids[0] == expert1_slot,
            "readmitted expert lost its frequency history");
    std::puts("PASS expert-cache ownership/protection/duplicates/ghost-history/bounds");
}

struct Fixture {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata{gguf_init_empty(), gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free};
    std::vector<std::vector<unsigned char>> payloads;
    ggml_type type;
    int hidden;
    bool mixed_types;
    std::uint64_t random = 918273645;

    explicit Fixture(ggml_type quant, bool mixed = false)
        : type(quant), hidden(ggml_blck_size(quant) > 32 ? 256 : 64), mixed_types(mixed) {
        require(metadata && tensors, "fixture allocation failed");
        auto* m = metadata.get();
        gguf_set_val_str(m, "general.architecture", "qwen3moe");
        gguf_set_val_str(m, "general.name", "SYNTHETIC-MOE-STREAMING-NOT-A-MODEL");
        const auto u32 = [&](const char* key, std::uint32_t value) {
            gguf_set_val_u32(m, (std::string("qwen3moe.") + key).c_str(), value);
        };
        u32("block_count", 4);
        u32("context_length", 256);
        u32("embedding_length", hidden);
        u32("feed_forward_length", hidden * 2);
        u32("expert_feed_forward_length", hidden);
        u32("expert_count", 16);
        u32("expert_used_count", 2);
        u32("attention.head_count", 4);
        u32("attention.head_count_kv", 2);
        u32("attention.key_length", hidden / 4);
        u32("attention.value_length", hidden / 4);
        u32("rope.dimension_count", hidden / 4);
        gguf_set_val_f32(m, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);
        gguf_set_val_f32(m, "qwen3moe.rope.freq_base", 10000000.0f);
        gguf_set_val_str(m, "tokenizer.ggml.model", "llama");
        std::vector<std::string> tokens{"<unk>", "<s>", "</s>"};
        for (int i = 3; i < 32; ++i)
            tokens.push_back("token" + std::to_string(i));
        std::vector<const char*> names;
        for (const auto& token : tokens)
            names.push_back(token.c_str());
        std::vector<float> scores(32, 0);
        std::vector<std::int32_t> types(32, 1);
        types[0] = 2;
        types[1] = types[2] = 3;
        gguf_set_arr_str(m, "tokenizer.ggml.tokens", names.data(), names.size());
        gguf_set_arr_data(m, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(),
                          scores.size());
        gguf_set_arr_data(m, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(),
                          types.size());
        gguf_set_val_u32(m, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(m, "tokenizer.ggml.eos_token_id", 2);
        gguf_set_val_u32(m, "tokenizer.ggml.unknown_token_id", 0);
    }

    void tensor(const std::string& name, int n0, int n1 = 1, int n2 = 1, bool quantize = false) {
        auto quant = quantize ? type : GGML_TYPE_F32;
        if (mixed_types && quantize) {
            if (name.find("attn_k.weight") != std::string::npos)
                quant = GGML_TYPE_Q8_0;
            else if (name.find("attn_output.weight") != std::string::npos)
                quant = GGML_TYPE_Q5_K;
            else if (name == "output.weight" ||
                     ((name.find("ffn_down_exps.weight") != std::string::npos ||
                       name.find("attn_v.weight") != std::string::npos) &&
                      (name.compare(0, 6, "blk.1.") == 0 || name.compare(0, 6, "blk.3.") == 0)))
                quant = GGML_TYPE_Q6_K;
        }
        auto* t = ggml_new_tensor_3d(tensors.get(), quant, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        std::vector<float> values(ggml_nelements(t));
        for (auto& value : values) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            value = n1 == 1 && n2 == 1 && name.find("norm") != std::string::npos
                        ? 1.0f
                        : (static_cast<float>((random >> 40) & 65535) / 32768.0f - 1) * 0.03f;
        }
        payloads.emplace_back(ggml_nbytes(t));
        if (quant == GGML_TYPE_F32)
            std::memcpy(payloads.back().data(), values.data(), payloads.back().size());
        else
            require(ggml_quantize_chunk(quant, values.data(), payloads.back().data(), 0, n1 * n2,
                                        n0, nullptr) == payloads.back().size(),
                    "fixture quantization failed");
        t->data = payloads.back().data();
        gguf_add_tensor(metadata.get(), t);
    }

    void write(const std::filesystem::path& path) {
        tensor("token_embd.weight", hidden, 32, 1, true);
        tensor("output.weight", hidden, 32, 1, true);
        tensor("output_norm.weight", hidden);
        for (int layer = 0; layer < 4; ++layer) {
            const auto p = "blk." + std::to_string(layer) + ".";
            tensor(p + "attn_norm.weight", hidden);
            tensor(p + "ffn_norm.weight", hidden);
            tensor(p + "attn_q.weight", hidden, hidden, 1, true);
            tensor(p + "attn_k.weight", hidden, hidden / 2, 1, true);
            tensor(p + "attn_v.weight", hidden, hidden / 2, 1, true);
            tensor(p + "attn_output.weight", hidden, hidden, 1, true);
            tensor(p + "attn_q_norm.weight", hidden / 4);
            tensor(p + "attn_k_norm.weight", hidden / 4);
            tensor(p + "ffn_gate_inp.weight", hidden, 16);
            tensor(p + "ffn_gate_exps.weight", hidden, hidden, 16, true);
            tensor(p + "ffn_up_exps.weight", hidden, hidden, 16, true);
            tensor(p + "ffn_down_exps.weight", hidden, hidden, 16, true);
        }
        require(gguf_write_to_file(metadata.get(), path.string().c_str(), false),
                "cannot write MoE fixture");
    }
};

void compare(const std::filesystem::path& path, std::size_t slots, bool per_node = false) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    params.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> reference(
        llama_model_load_from_file(path.string().c_str(), params), llama_model_free);
    require(reference != nullptr, "resident MoE fixture load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 4;
    cp.n_ubatch = 1; // Match physical ubatches for an arithmetic, not batching, comparison.
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(reference.get(), cp), llama_free);
    require(context != nullptr, "resident MoE fixture context failed");
    MoeStreamingModel stream(path.string(), 64 * 1024 * 1024, slots, per_node);
    stream.create_context(64, 4, 2);
    rejects([&] { (void)stream.logits(); }, "uninitialized MoE logits exposed");
    require(stream.stats().expert_read_bytes == 0, "experts were eagerly read during load");
    float max_error = 0;
    for (int step_number = 0; step_number < 12; ++step_number) {
        llama_token tokens[] = {3 + step_number, 4 + step_number, 5 + step_number, 6 + step_number};
        const int count = step_number == 0 ? 4 : step_number == 3 ? 2 : 1;
        const auto batch = llama_batch_get_one(tokens, count);
        require(llama_decode(context.get(), batch) == 0, "resident MoE decode failed");
        stream.decode(batch);
        const auto* expected = llama_get_logits(context.get());
        const auto* actual = stream.logits();
        for (int token = 0; token < 32; ++token) {
            require(std::isfinite(expected[token]) && std::isfinite(actual[token]),
                    "non-finite MoE logit");
            max_error = std::max(max_error, std::abs(expected[token] - actual[token]));
            require(std::abs(expected[token] - actual[token]) <=
                        1e-5f + 1e-5f * std::abs(expected[token]),
                    "MoE resident/streamed logit mismatch");
        }
    }
    const auto& stats = stream.stats();
    require(stats.common_bytes + stats.expert_cache_bytes < stats.logical_weight_bytes,
            "MoE fixture did not reduce allocated weight storage");
    require(stats.cache.hits > 0 && stats.cache.evictions > 0 && stats.cache.completed_loads > 0,
            "MoE fixture did not exercise hits, misses and evictions");
    require(stats.cache.misses == stats.cache.completed_loads &&
                stats.cache.hits + stats.cache.misses == 2 * stats.cache.decisions,
            "MoE cache counters counted projections instead of routing decisions");
    require(stats.synchronized_checkpoints <= stats.chunk_visits,
            "MoE synchronization mode did not match its contract");
    std::printf(
        "PASS %s slots=%zu per_node=%d max_logit_error=%g hits=%llu misses=%llu evictions=%llu "
        "logical_weights=%zu allocated_weights=%zu expert_read_bytes=%llu checkpoints=%llu "
        "chunks=%llu\n",
        path.filename().string().c_str(), slots, per_node ? 1 : 0, max_error,
        static_cast<unsigned long long>(stats.cache.hits),
        static_cast<unsigned long long>(stats.cache.misses),
        static_cast<unsigned long long>(stats.cache.evictions), stats.logical_weight_bytes,
        stats.common_bytes + stats.expert_cache_bytes,
        static_cast<unsigned long long>(stats.expert_read_bytes),
        static_cast<unsigned long long>(stats.synchronized_checkpoints),
        static_cast<unsigned long long>(stats.chunk_visits));
    rejects([&] { MoeStreamingModel invalid(path.string(), 1, slots); },
            "small MoE budget accepted");
    rejects([&] { MoeStreamingModel invalid(path.string(), 64 * 1024 * 1024, 1); },
            "slots below top-k accepted");
    {
        MoeStreamingModel cancelled(path.string(), 64 * 1024 * 1024, slots);
        cancelled.create_context(64, 4, 2);
        cancelled.cancel();
        llama_token token = 3;
        rejects([&] { cancelled.decode(llama_batch_get_one(&token, 1)); },
                "cancelled MoE decode accepted");
        require(cancelled.stats().expert_read_bytes == 0, "cancelled MoE decode read experts");
    }
    // Bytes in an already cached expert may remain valid after truncation, so
    // force subsequent tokens to request misses and reject the first short read.
    std::filesystem::resize_file(path, 64);
    bool failed = false;
    for (llama_token token = 3; token < 32 && !failed; ++token) {
        try {
            stream.decode(llama_batch_get_one(&token, 1));
        } catch (const std::exception&) {
            failed = true;
        }
    }
    require(failed, "truncated MoE file never caused a cache-miss read failure");
    rejects([&] { (void)stream.logits(); }, "failed MoE decode exposed logits");
    llama_token token = 3;
    rejects([&] { stream.decode(llama_batch_get_one(&token, 1)); },
            "poisoned MoE context was reused");
}
} // namespace

int main() {
    llama_backend_init();
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory =
        std::filesystem::temp_directory_path() / ("xllama-moe-smoke-" + std::to_string(id));
    int result = 0;
    try {
        cache_tests();
        require(std::filesystem::create_directory(directory),
                "cannot create MoE fixture directory");
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
            for (const std::size_t slots : {2, 4}) {
                const auto path = directory / (std::string("qwen3moe-") + ggml_type_name(type) +
                                               "-" + std::to_string(slots) + ".gguf");
                {
                    Fixture fixture(type);
                    fixture.write(path);
                }
                compare(path, slots);
                std::filesystem::remove(path);
            }
        }
        {
            const auto path = directory / "qwen3moe-q4_K-per-node.gguf";
            {
                Fixture fixture(GGML_TYPE_Q4_K);
                fixture.write(path);
            }
            compare(path, 4, true);
            std::filesystem::remove(path);
        }
        {
            const auto path = directory / "qwen3moe-mixed-k.gguf";
            {
                Fixture fixture(GGML_TYPE_Q4_K, true);
                fixture.write(path);
            }
            compare(path, 4);
            std::filesystem::remove(path);
        }
        std::filesystem::remove(directory);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MoE streaming smoke: %s\nFixtures retained at %s\n",
                     error.what(), directory.string().c_str());
        result = 1;
    }
    llama_backend_free();
    return result;
}
