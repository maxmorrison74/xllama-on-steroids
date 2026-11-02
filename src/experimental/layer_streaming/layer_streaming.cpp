// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "layer_streaming.h"

// Deliberate research-only dependency; do not expose this through the shared API.
#include "ggml-backend.h"
#include "gguf.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace xllama::experimental {
namespace {

std::size_t add_checked(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::runtime_error("streaming size overflow");
    return a + b;
}

std::size_t aligned(std::size_t size) {
    return add_checked(size, 63) & ~std::size_t(63);
}

int block_index(const std::string& name) {
    if (name.compare(0, 4, "blk.") != 0)
        return -1;
    const auto end = name.find('.', 4);
    if (end == std::string::npos || end == 4 || end - 4 > 6)
        throw std::runtime_error("invalid block tensor name: " + name);
    int result = 0;
    for (std::size_t i = 4; i < end; ++i) {
        if (name[i] < '0' || name[i] > '9')
            throw std::runtime_error("invalid block tensor name: " + name);
        result = result * 10 + name[i] - '0';
    }
    return result;
}

struct FileCloser {
    void operator()(FILE* file) const {
        if (file)
            std::fclose(file);
    }
};
struct GgufCloser {
    void operator()(gguf_context* ctx) const {
        if (ctx)
            gguf_free(ctx);
    }
};
struct BufferCloser {
    void operator()(ggml_backend_buffer* buffer) const {
        if (buffer)
            ggml_backend_buffer_free(buffer);
    }
};
struct ModelCloser {
    void operator()(llama_model* model) const {
        if (model)
            llama_model_free(model);
    }
};
struct ContextCloser {
    void operator()(llama_context* ctx) const {
        if (ctx)
            llama_free(ctx);
    }
};
using Buffer = std::unique_ptr<ggml_backend_buffer, BufferCloser>;

Buffer allocate(std::size_t size) {
    Buffer buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size));
    if (!buffer)
        throw std::runtime_error("cannot allocate bounded streaming weight buffer");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

void seek(FILE* file, std::size_t offset) {
    if (offset > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::runtime_error("streaming file offset overflow");
#ifdef _WIN32
    const int result = _fseeki64(file, static_cast<std::int64_t>(offset), SEEK_SET);
#else
    const int result = fseeko(file, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (result != 0)
        throw std::runtime_error("streaming file seek failed");
}

} // namespace

struct LayerStreamingModel::Impl {
    struct Weight {
        ggml_tensor* tensor;
        std::size_t file_offset;
        std::size_t bytes;
        std::size_t buffer_offset;
        int block;
    };

    std::unique_ptr<FILE, FileCloser> file;
    // Destruction order: context, model, then backing buffers and file.
    Buffer resident;
    std::array<Buffer, 2> slots;
    std::unique_ptr<llama_model, ModelCloser> loaded;
    std::unique_ptr<llama_context, ContextCloser> context;
    std::vector<Weight> weights;
    std::map<int, std::vector<std::size_t>> blocks;
    std::unordered_map<const ggml_tensor*, int> tensor_blocks;
    std::array<int, 2> slot_blocks{{-1, -1}};
    StreamingStats stats;
    bool poisoned = false;
    bool has_logits = false;
    std::atomic<bool> cancelled{false};

    ~Impl() {
        context.reset();
        loaded.reset();
    }

    void read(const Weight& weight) {
        seek(file.get(), weight.file_offset);
        auto* dst = static_cast<unsigned char*>(weight.tensor->data);
        std::size_t done = 0;
        // Bound individual reads, including the resident embedding/output loads.
        while (done < weight.bytes) {
            if (cancelled.load(std::memory_order_relaxed))
                throw std::runtime_error("streaming cancelled");
            const auto count = std::min<std::size_t>(16 * 1024 * 1024, weight.bytes - done);
            if (std::fread(dst + done, 1, count, file.get()) != count)
                throw std::runtime_error("short streaming read: " +
                                         std::string(weight.tensor->name));
            done += count;
            stats.bytes_read += count;
        }
    }

    int source_block(const ggml_tensor* tensor) const {
        // Views of weights retain the same backing addresses as their roots.
        for (int depth = 0; tensor && depth < 32; ++depth) {
            const auto found = tensor_blocks.find(tensor);
            if (found != tensor_blocks.end())
                return found->second;
            if (!tensor->view_src)
                return -1;
            tensor = tensor->view_src;
        }
        if (tensor)
            throw std::runtime_error("excessive streaming tensor view depth");
        return -1;
    }

    void prepare(ggml_tensor* node) {
        if (cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("streaming cancelled");
        std::array<int, 2> required{{-1, -1}};
        for (const auto* source : node->src) {
            const int block = source_block(source);
            if (block < 0)
                continue;
            auto& needed = required[block % 2];
            if (needed != -1 && needed != block)
                throw std::runtime_error("node requires conflicting streaming weight slots");
            needed = block;
        }
        for (std::size_t slot = 0; slot < required.size(); ++slot) {
            const int block = required[slot];
            if (block < 0 || slot_blocks[slot] == block)
                continue;
            slot_blocks[slot] = -1;
            for (const auto index : blocks.at(block))
                read(weights[index]);
            slot_blocks[slot] = block;
            ++stats.layer_loads;
        }
    }

    static bool evaluate(ggml_tensor* node, bool ask, void* userdata) {
        auto& self = *static_cast<Impl*>(userdata);
        // Returning true for EVERY node prevents scheduler look-ahead from
        // overwriting a slot before earlier nodes have finished. The pinned
        // scheduler synchronizes after each node. This is synchronous streaming,
        // not asynchronous prefetch, and intentionally sacrifices performance.
        // This callback runs on the C++ scheduler caller, not a worker thread.
        // Exceptions unwind before graph_compute; decode poisons this context.
        if (ask)
            self.prepare(node);
        return true;
    }
};

LayerStreamingModel::LayerStreamingModel(const std::string& path, std::size_t budget)
    : impl_(std::make_unique<Impl>()) {
    auto& self = *impl_;
    self.file.reset(std::fopen(path.c_str(), "rb"));
    if (!self.file)
        throw std::runtime_error("cannot open streaming GGUF");
    if (std::setvbuf(self.file.get(), nullptr, _IONBF, 0) != 0)
        throw std::runtime_error("cannot disable stdio read-ahead for streaming");
    const gguf_init_params init{true, nullptr};
    std::unique_ptr<gguf_context, GgufCloser> metadata(gguf_init_from_file(path.c_str(), init));
    if (!metadata)
        throw std::runtime_error("cannot read streaming GGUF metadata");
    const auto split = gguf_find_key(metadata.get(), "split.count");
    if (split >= 0)
        throw std::runtime_error("split GGUF streaming is not implemented in this host spike");
    const auto arch_key = gguf_find_key(metadata.get(), "general.architecture");
    if (arch_key < 0 || gguf_get_kv_type(metadata.get(), arch_key) != GGUF_TYPE_STRING)
        throw std::runtime_error("missing GGUF architecture");
    const std::string arch = gguf_get_val_str(metadata.get(), arch_key);
    if (arch != "llama" && arch != "qwen35")
        throw std::runtime_error("streaming spike only accepts dense llama/qwen35");

    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    params.use_extra_bufts = false;
    params.no_alloc = true;
    params.load_mtp = false;
    self.loaded.reset(llama_model_load_from_file(path.c_str(), params));
    if (!self.loaded)
        throw std::runtime_error("cannot create metadata-only streaming model");
    if (self.loaded->hparams.n_expert != 0)
        throw std::runtime_error("MoE is not supported by this streaming spike");

    std::map<int, std::size_t> block_sizes;
    for (const auto& [name, tensor] : self.loaded->tensors_by_name) {
        const auto id = gguf_find_tensor(metadata.get(), name.c_str());
        if (id < 0 || tensor->view_src ||
            tensor->type != gguf_get_tensor_type(metadata.get(), id) ||
            ggml_nbytes(tensor) != gguf_get_tensor_size(metadata.get(), id))
            throw std::runtime_error("unsupported virtual/transformed streaming tensor: " + name);
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] != gguf_get_tensor_ne(metadata.get(), id)[dim])
                throw std::runtime_error("streaming tensor shape mismatch: " + name);
        }
        const int block = block_index(name);
        auto& offset = block < 0 ? self.stats.resident_bytes : block_sizes[block];
        const auto size = ggml_nbytes(tensor);
        self.weights.push_back({tensor,
                                add_checked(gguf_get_data_offset(metadata.get()),
                                            gguf_get_tensor_offset(metadata.get(), id)),
                                size, offset, block});
        offset = add_checked(offset, aligned(size));
        self.stats.total_weight_bytes = add_checked(self.stats.total_weight_bytes, size);
        if (block >= 0) {
            self.blocks[block].push_back(self.weights.size() - 1);
            self.tensor_blocks.emplace(tensor, block);
        }
    }
    if (self.blocks.size() != self.loaded->hparams.n_layer())
        throw std::runtime_error("streaming model has incomplete decoder blocks");
    int expected = 0;
    for (const auto& [block, size] : block_sizes) {
        if (block != expected++)
            throw std::runtime_error("streaming decoder blocks must be contiguous");
        self.stats.slot_bytes = std::max(self.stats.slot_bytes, size);
    }
    const auto required = add_checked(self.stats.resident_bytes,
                                      add_checked(self.stats.slot_bytes, self.stats.slot_bytes));
    if (required > budget)
        throw std::runtime_error("streaming weight buffers exceed explicit budget");
    self.resident = allocate(self.stats.resident_bytes);
    for (auto& slot : self.slots)
        slot = allocate(self.stats.slot_bytes);
    for (const auto& weight : self.weights) {
        auto* buffer = weight.block < 0 ? self.resident.get() : self.slots[weight.block % 2].get();
        auto* address = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buffer)) +
                        weight.buffer_offset;
        // no_alloc used dummy weight buffers; replace them only after budget admission.
        weight.tensor->buffer = nullptr;
        if (ggml_backend_tensor_alloc(buffer, weight.tensor, address) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("cannot attach streaming tensor buffer");
        if (weight.block < 0)
            self.read(weight);
    }
    // This internal promotion is why the adapter is pinned and host-only. A
    // production/UWP integration needs a supported loader ownership contract.
    self.loaded->hparams.no_alloc = false;
}

LayerStreamingModel::~LayerStreamingModel() = default;

void LayerStreamingModel::create_context(std::uint32_t n_ctx, std::uint32_t batch_size,
                                         int threads) {
    auto& self = *impl_;
    if (self.context || self.poisoned || self.cancelled.load(std::memory_order_relaxed) ||
        n_ctx == 0 || batch_size == 0 || threads <= 0)
        throw std::runtime_error("invalid/repeated streaming context creation");
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = batch_size;
    params.n_ubatch = batch_size;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.offload_kqv = false;
    params.op_offload = false;
    params.cb_eval = &Impl::evaluate;
    params.cb_eval_user_data = &self;
    self.context.reset(llama_init_from_model(self.loaded.get(), params));
    if (!self.context)
        throw std::runtime_error("cannot create streaming context");
}

void LayerStreamingModel::decode(llama_batch batch) {
    auto& self = *impl_;
    if (!self.context || self.poisoned)
        throw std::runtime_error("streaming context unavailable or poisoned");
    self.has_logits = false;
    try {
        if (self.cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("streaming cancelled");
        if (batch.n_tokens <= 0 || !batch.token || batch.embd)
            throw std::runtime_error("streaming spike requires a non-empty token batch");
        if (batch.logits && std::none_of(batch.logits, batch.logits + batch.n_tokens,
                                         [](int8_t enabled) { return enabled != 0; }))
            throw std::runtime_error("streaming batch must request logits");
        if (llama_decode(self.context.get(), batch) != 0)
            throw std::runtime_error("streaming decode failed");
        if (self.cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("streaming cancelled");
        self.has_logits = true;
    } catch (...) {
        self.poisoned = true;
        throw;
    }
}

const float* LayerStreamingModel::logits() const {
    if (!impl_->context || impl_->poisoned || !impl_->has_logits ||
        impl_->cancelled.load(std::memory_order_relaxed))
        throw std::runtime_error("no valid streaming logits");
    return llama_get_logits(impl_->context.get());
}

void LayerStreamingModel::cancel() noexcept {
    impl_->cancelled.store(true, std::memory_order_relaxed);
}

const llama_model* LayerStreamingModel::model() const {
    return impl_->loaded.get();
}
const StreamingStats& LayerStreamingModel::stats() const {
    return impl_->stats;
}

} // namespace xllama::experimental
