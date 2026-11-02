// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
// Synthetic fixtures produce noise, not a usable model. No downloaded weights.
#include "layer_streaming.h"

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
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Fixture {
    gguf_context* metadata = gguf_init_empty();
    ggml_context* tensors = ggml_init({4 * 1024 * 1024, nullptr, true});
    std::vector<std::vector<unsigned char>> payloads;
    ggml_type matrix_type;
    int hidden;

    Fixture(const std::string& arch, ggml_type type)
        : matrix_type(type), hidden(ggml_blck_size(type) > 32 ? 256 : 64) {
        require(metadata && tensors, "fixture metadata allocation failed");
        gguf_set_val_str(metadata, "general.architecture", arch.c_str());
        gguf_set_val_str(metadata, "general.name", "SYNTHETIC-STREAMING-TEST-NOT-A-MODEL");
        const auto u32 = [&](const char* key, std::uint32_t value) {
            gguf_set_val_u32(metadata, (arch + "." + key).c_str(), value);
        };
        u32("block_count", 4);
        u32("context_length", 256);
        u32("embedding_length", hidden);
        u32("feed_forward_length", hidden * 2);
        u32("attention.head_count", 4);
        u32("attention.head_count_kv", 2);
        u32("attention.key_length", hidden / 4);
        u32("attention.value_length", hidden / 4);
        u32("rope.dimension_count", hidden / 4);
        gguf_set_val_f32(metadata, (arch + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
        if (arch == "qwen35") {
            const std::int32_t sections[] = {hidden / 32, hidden / 32, hidden / 16, 0};
            gguf_set_arr_data(metadata, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections,
                              4);
            u32("full_attention_interval", 4);
            u32("ssm.conv_kernel", 4);
            u32("ssm.inner_size", hidden);
            u32("ssm.state_size", hidden / 4);
            u32("ssm.time_step_rank", 4);
            u32("ssm.group_count", 2);
        }
        gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
        std::vector<std::string> tokens;
        std::vector<const char*> names;
        std::vector<float> scores(32, 0);
        std::vector<std::int32_t> types(32, 1);
        tokens = {"<unk>", "<s>", "</s>"};
        for (int i = 3; i < 32; ++i)
            tokens.push_back("token" + std::to_string(i));
        for (const auto& token : tokens)
            names.push_back(token.c_str());
        types[0] = 2;
        types[1] = types[2] = 3;
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", names.data(), names.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(),
                          scores.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(),
                          types.size());
        gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
        gguf_set_val_u32(metadata, "tokenizer.ggml.unknown_token_id", 0);
    }

    ~Fixture() {
        gguf_free(metadata);
        ggml_free(tensors);
    }

    void tensor(const std::string& name, int first, int second = 0, bool quantized = false) {
        const auto type = quantized && second > 0 ? matrix_type : GGML_TYPE_F32;
        ggml_tensor* tensor = second > 0 ? ggml_new_tensor_2d(tensors, type, first, second)
                                         : ggml_new_tensor_1d(tensors, type, first);
        ggml_set_name(tensor, name.c_str());
        std::vector<float> values(ggml_nelements(tensor));
        const auto seed = static_cast<float>(payloads.size() + 1);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = second == 0 && name.find("norm") != std::string::npos
                            ? 1.0f
                            : 0.03f * std::sin((i + 1) * (seed + 1) * 0.17f);
        payloads.emplace_back(ggml_nbytes(tensor));
        if (type == GGML_TYPE_F32) {
            std::memcpy(payloads.back().data(), values.data(), payloads.back().size());
        } else {
            const auto written = ggml_quantize_chunk(type, values.data(), payloads.back().data(), 0,
                                                     second, first, nullptr);
            require(written == payloads.back().size(), "fixture quantization size mismatch");
        }
        tensor->data = payloads.back().data();
        gguf_add_tensor(metadata, tensor);
    }

    void write(const std::string& arch, bool quantized, const std::filesystem::path& path) {
        tensor("token_embd.weight", hidden, 32, quantized);
        tensor("output_norm.weight", hidden);
        tensor("output.weight", hidden, 32, quantized);
        for (int layer = 0; layer < 4; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            tensor(p + "attn_norm.weight", hidden);
            if (arch == "qwen35")
                tensor(p + "post_attention_norm.weight", hidden);
            else
                tensor(p + "ffn_norm.weight", hidden);
            if (arch == "qwen35" && layer < 3) {
                tensor(p + "attn_qkv.weight", hidden, hidden * 2, quantized);
                tensor(p + "attn_gate.weight", hidden, hidden, quantized);
                tensor(p + "ssm_conv1d.weight", 4, hidden * 2);
                tensor(p + "ssm_dt.bias", 4);
                tensor(p + "ssm_a", 4);
                tensor(p + "ssm_beta.weight", hidden, 4, quantized);
                tensor(p + "ssm_alpha.weight", hidden, 4, quantized);
                tensor(p + "ssm_norm.weight", hidden / 4);
                tensor(p + "ssm_out.weight", hidden, hidden, quantized);
            } else {
                tensor(p + "attn_q.weight", hidden, arch == "qwen35" ? hidden * 2 : hidden,
                       quantized);
                tensor(p + "attn_k.weight", hidden, hidden / 2, quantized);
                tensor(p + "attn_v.weight", hidden, hidden / 2, quantized);
                tensor(p + "attn_output.weight", hidden, hidden, quantized);
                if (arch == "qwen35") {
                    tensor(p + "attn_q_norm.weight", hidden / 4);
                    tensor(p + "attn_k_norm.weight", hidden / 4);
                }
            }
            tensor(p + "ffn_gate.weight", hidden, hidden * 2, quantized);
            tensor(p + "ffn_up.weight", hidden, hidden * 2, quantized);
            tensor(p + "ffn_down.weight", hidden * 2, hidden, quantized);
        }
        require(gguf_write_to_file(metadata, path.string().c_str(), false),
                "cannot write synthetic GGUF");
    }
};

struct ModelCloser {
    void operator()(llama_model* p) const {
        llama_model_free(p);
    }
};
struct ContextCloser {
    void operator()(llama_context* p) const {
        llama_free(p);
    }
};

void compare(const std::filesystem::path& path) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    params.load_mtp = false;
    std::unique_ptr<llama_model, ModelCloser> reference(
        llama_model_load_from_file(path.string().c_str(), params));
    require(reference != nullptr, "resident fixture load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = cp.n_ubatch = 4;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.offload_kqv = cp.op_offload = false;
    std::unique_ptr<llama_context, ContextCloser> context(
        llama_init_from_model(reference.get(), cp));
    require(context != nullptr, "resident fixture context failed");
    xllama::experimental::LayerStreamingModel streamed(path.string(), 16 * 1024 * 1024);
    streamed.create_context(64, 4, 2);
    bool rejected = false;
    try {
        (void)streamed.logits();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "uninitialized streaming logits were exposed");
    const auto& stats = streamed.stats();
    require(stats.resident_bytes + 2 * stats.slot_bytes < stats.total_weight_bytes,
            "streaming fixture did not reduce allocated weight storage");
    float max_error = 0;
    for (int step = 0; step < 4; ++step) {
        llama_token tokens[] = {3, 4, 5, 6};
        const int count = step == 0 ? 4 : step == 2 ? 2 : 1;
        const auto batch = llama_batch_get_one(tokens, count);
        require(llama_decode(context.get(), batch) == 0, "resident fixture decode failed");
        streamed.decode(batch);
        const auto* a = llama_get_logits(context.get());
        const auto* b = streamed.logits();
        for (int i = 0; i < 32; ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite fixture logit");
            max_error = std::max(max_error, std::abs(a[i] - b[i]));
            require(std::abs(a[i] - b[i]) <= 1e-5f + 1e-5f * std::abs(a[i]),
                    "resident/streamed logit mismatch");
        }
    }
    require(stats.layer_loads >= 16, "streaming did not evict/reload decoder layers");
    std::printf(
        "PASS %s max_logit_error=%g layer_loads=%llu weight_bytes=%zu resident_and_slots=%zu\n",
        path.filename().string().c_str(), max_error,
        static_cast<unsigned long long>(stats.layer_loads), stats.total_weight_bytes,
        stats.resident_bytes + 2 * stats.slot_bytes);
    rejected = false;
    try {
        xllama::experimental::LayerStreamingModel too_small(path.string(), 1);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "insufficient weight budget was accepted");
    {
        xllama::experimental::LayerStreamingModel cancelled(path.string(), 16 * 1024 * 1024);
        cancelled.create_context(64, 4, 2);
        cancelled.cancel();
        llama_token token = 3;
        rejected = false;
        try {
            cancelled.decode(llama_batch_get_one(&token, 1));
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected && cancelled.stats().layer_loads == 0,
                "cancelled streaming decode read layer weights");
    }
    std::filesystem::resize_file(path, 64);
    llama_token next = 7;
    rejected = false;
    try {
        streamed.decode(llama_batch_get_one(&next, 1));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "truncated streaming file was accepted");
    rejected = false;
    try {
        (void)streamed.logits();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "failed streaming decode exposed logits");
}
} // namespace

int main() {
    llama_backend_init();
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir =
        std::filesystem::temp_directory_path() / ("xllama-streaming-smoke-" + std::to_string(id));
    int result = 0;
    try {
        require(std::filesystem::create_directory(dir), "cannot create fixture directory");
        for (const auto* arch : {"llama", "qwen35"}) {
            for (const auto type :
                 {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
                const auto path = dir / (std::string(arch) + "-" + ggml_type_name(type) + ".gguf");
                {
                    Fixture fixture(arch, type);
                    fixture.write(arch, type != GGML_TYPE_F32, path);
                }
                compare(path);
                std::filesystem::remove(path);
            }
        }
        std::filesystem::remove(dir);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL streaming smoke: %s\nFixtures retained at %s\n", error.what(),
                     dir.string().c_str());
        result = 1;
    }
    llama_backend_free();
    return result;
}
