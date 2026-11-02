// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "moe_streaming.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace xllama::experimental {
namespace {
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::runtime_error("MoE size overflow");
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::runtime_error("MoE size overflow");
    return a * b;
}

std::size_t align64(std::size_t size) {
    return checked_add(size, 63) & ~std::size_t(63);
}

Buffer allocate_weights(std::size_t bytes) {
    Buffer result(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes),
                  ggml_backend_buffer_free);
    if (!result)
        throw std::runtime_error("cannot allocate bounded MoE weight storage");
    ggml_backend_buffer_set_usage(result.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

int expert_block(const std::string& name, unsigned& projection) {
    projection = 0;
    if (name.find("_exps") == std::string::npos)
        return -1;
    const auto end = name.find('.', 4);
    if (name.compare(0, 4, "blk.") || end == std::string::npos || end == 4 || end > 10)
        throw std::runtime_error("invalid routed-expert tensor name");
    const auto suffix = name.substr(end + 1);
    if (suffix == "ffn_gate_exps.weight")
        projection = 1;
    else if (suffix == "ffn_up_exps.weight")
        projection = 2;
    else if (suffix == "ffn_down_exps.weight")
        projection = 4;
    else
        throw std::runtime_error("unsupported routed-expert projection: " + name);
    int block = 0;
    for (std::size_t i = 4; i < end; ++i) {
        if (name[i] < '0' || name[i] > '9')
            throw std::runtime_error("invalid routed-expert block index");
        block = block * 10 + name[i] - '0';
    }
    return block;
}

std::vector<std::int32_t> read_ids(const ggml_tensor* ids) {
    if (!ids || ids->type != GGML_TYPE_I32 || !ids->data || !ids->buffer ||
        !ggml_backend_buffer_is_host(ids->buffer))
        throw std::runtime_error("MoE spike requires host router IDs");
    const std::size_t n =
        static_cast<std::size_t>(ids->ne[0]) * static_cast<std::size_t>(ids->ne[1]);
    if (n == 0 || ids->nb[0] != sizeof(std::int32_t))
        throw std::runtime_error("MoE spike requires contiguous int32 router IDs");
    std::vector<std::int32_t> out(n);
    const auto* src = static_cast<const unsigned char*>(ids->data);
    for (std::size_t i = 0; i < n; ++i)
        std::memcpy(&out[i], src + i * ids->nb[0], sizeof(std::int32_t));
    return out;
}

void write_ids(ggml_tensor* ids, const std::vector<std::int32_t>& values) {
    if (!ids || !ids->data || ids->nb[0] != sizeof(std::int32_t))
        throw std::runtime_error("MoE spike cannot write router IDs");
    const std::size_t n =
        static_cast<std::size_t>(ids->ne[0]) * static_cast<std::size_t>(ids->ne[1]);
    if (values.size() != n)
        throw std::runtime_error("MoE spike router ID count mismatch");
    auto* dst = static_cast<unsigned char*>(ids->data);
    for (std::size_t i = 0; i < n; ++i)
        std::memcpy(dst + i * ids->nb[0], &values[i], sizeof(std::int32_t));
}
} // namespace

struct MoeStreamingModel::Impl {
    struct Weight {
        ggml_tensor* tensor;
        std::size_t file_offset;
        std::size_t original_bytes;
        std::size_t slab_bytes;
        std::size_t buffer_offset;
        int block;
        unsigned projection;
    };
    struct Layer {
        Layer(std::size_t experts, std::size_t slots) : cache(experts, slots) {}
        ExpertCache cache;
        ExpertPlan plan;
        std::vector<std::size_t> weights;
        std::vector<std::int32_t> requested_ids;
        Buffer storage{nullptr, ggml_backend_buffer_free};
        std::size_t bytes = 0;
        unsigned projection_mask = 0;
    };

    MoeFile file;
    Buffer common{nullptr, ggml_backend_buffer_free};
    std::map<int, Layer> layers;
    std::vector<Weight> weights;
    std::unordered_map<const ggml_tensor*, std::size_t> expert_weights;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> loaded{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    MoeStreamingStats stats;
    std::size_t experts = 0;
    std::size_t active_experts = 0;
    std::size_t slots = 0;
    std::size_t physical_ubatch = 1;
    int active_layer = -1;
    ggml_tensor* pending_node = nullptr;
    ggml_tensor* pending_ids = nullptr;
    std::vector<std::int32_t> original_ids;
    unsigned pending_projection = 0;
    bool poisoned = false;
    bool has_logits = false;
    std::atomic<bool> cancelled{false};

    ~Impl() {
        context.reset();
        loaded.reset();
    }

    void read(std::size_t offset, void* destination, std::size_t bytes, bool expert) {
        if (bytes == 0)
            return;
        bool unbuffered = false;
        if (!file.valid()) {
            bool opened = file.open(path_, true, unbuffered);
            if (!opened)
                throw std::runtime_error("cannot open MoE GGUF for positioned reads");
            stats.file_unbuffered = unbuffered;
        }
        const std::size_t got = file.read(offset, destination, bytes);
        if (got != bytes)
            throw std::runtime_error("short MoE streaming read");
        (expert ? stats.expert_read_bytes : stats.common_read_bytes) += bytes;
    }

    // Path used by the constructor before `impl_` is fully formed.
    std::string path_;

    void restore_ids() noexcept {
        if (pending_ids) {
            write_ids(pending_ids, original_ids);
            pending_node = nullptr;
            pending_ids = nullptr;
        }
    }

    void snapshot_stats() noexcept {
        stats.cache = {};
        for (const auto& item : layers) {
            const auto& value = item.second.cache.stats();
            stats.cache.decisions += value.decisions;
            stats.cache.hits += value.hits;
            stats.cache.misses += value.misses;
            stats.cache.completed_loads += value.completed_loads;
            stats.cache.evictions += value.evictions;
        }
    }

    void load_experts(int block, const ExpertPlan& plan) {
        auto& layer = layers.at(block);
        for (const auto& binding : plan.bindings) {
            if (!binding.needs_read)
                continue;
            for (const auto index : layer.weights) {
                const auto& weight = weights[index];
                const auto source =
                    checked_add(weight.file_offset, checked_mul(binding.expert, weight.slab_bytes));
                auto* target = static_cast<unsigned char*>(weight.tensor->data) +
                               checked_mul(binding.slot, weight.slab_bytes);
                read(source, target, weight.slab_bytes, true);
            }
            // Valid only once every projection of this layer for this step has been
            // read; mark_loaded is called after the final projection completes.
            layer.cache.mark_loaded(binding.slot);
        }
    }

    static bool evaluate(ggml_tensor* node, bool ask, void* userdata) {
        auto& self = *static_cast<Impl*>(userdata);
        if (ask)
            return self.before(node);
        self.after(node);
        return true;
    }

    bool before(ggml_tensor* node) {
        ++stats.chunk_visits;
        if (cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("MoE streaming cancelled");
        if (pending_ids)
            throw std::runtime_error("previous MoE node has not synchronized");
        if (ggml_nelements(node) == 0)
            return true;
        if (node->op != GGML_OP_MUL_MAT_ID)
            return true;
        const auto index = expert_source(node->src[0]);
        if (index == weights.size()) {
            for (const auto* source : node->src) {
                if (expert_source(source) != weights.size())
                    throw std::runtime_error("unexpected streamed expert operand");
            }
            return true;
        }
        if (node->src[0] != weights[index].tensor || node->op != GGML_OP_MUL_MAT_ID)
            throw std::runtime_error("streamed expert used outside MUL_MAT_ID");
        const auto& weight = weights[index];
        auto* ids = node->src[2];
        const auto all_ids = read_ids(ids);
        const std::size_t n_tokens = static_cast<std::size_t>(ids->ne[1]);
        auto& layer = layers.at(weight.block);
        const bool new_step =
            active_layer < 0 || active_block_ != weight.block || layer.requested_ids != all_ids;
        if (new_step) {
            restore_ids();
            layer.cache.plan(all_ids, layer.plan);
            layer.requested_ids = all_ids;
            active_block_ = weight.block;
            active_layer = weight.block;
            load_experts(weight.block, layer.plan);
        }
        // Translate selected expert IDs to cache slots for this matmul only.
        std::vector<std::int32_t> remapped(all_ids.size());
        for (std::size_t i = 0; i < all_ids.size(); ++i)
            remapped[i] = static_cast<std::int32_t>(layer.cache.slot_for(all_ids[i]));
        original_ids = all_ids;
        pending_node = node;
        pending_ids = ids;
        pending_projection = weight.projection;
        write_ids(ids, remapped);
        ++stats.chunk_visits;
        return true;
    }

    void after(ggml_tensor* node) {
        ++stats.synchronized_checkpoints;
        if (!pending_ids)
            return;
        if (node != pending_node)
            throw std::runtime_error("MoE callback synchronization mismatch");
        restore_ids();
        auto& layer = layers.at(active_layer);
        layer.cache.finish();
        active_layer = -1;
    }

    std::size_t expert_source(const ggml_tensor* source) const {
        for (int depth = 0; source && depth < 32; ++depth) {
            const auto found = expert_weights.find(source);
            if (found != expert_weights.end())
                return found->second;
            if (!source->view_src)
                return weights.size();
            source = source->view_src;
        }
        if (source)
            throw std::runtime_error("excessive MoE weight-view depth");
        return weights.size();
    }

    int active_block_ = -1;
};

MoeStreamingModel::MoeStreamingModel(const std::string& path, std::size_t budget, std::size_t slots,
                                     std::size_t physical_ubatch)
    : impl_(std::make_unique<Impl>()) {
    auto& self = *impl_;
    self.path_ = path;
    self.physical_ubatch = physical_ubatch == 0 ? 1 : physical_ubatch;
    bool unbuffered = false;
    if (!self.file.open(path, true, unbuffered))
        throw std::runtime_error("cannot open MoE streaming GGUF for positioned reads");
    self.stats.file_unbuffered = unbuffered;
    const gguf_init_params init{true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.c_str(), init), gguf_free);
    if (!metadata || gguf_find_key(metadata.get(), "split.count") >= 0)
        throw std::runtime_error("MoE spike requires an unsplit GGUF");
    const auto arch = gguf_find_key(metadata.get(), "general.architecture");
    if (arch < 0 || gguf_get_kv_type(metadata.get(), arch) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata.get(), arch)) != "qwen3moe")
        throw std::runtime_error("MoE spike only accepts qwen3moe");
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    params.use_extra_bufts = false;
    params.no_alloc = true;
    params.load_mtp = false;
    self.loaded.reset(llama_model_load_from_file(path.c_str(), params));
    if (!self.loaded)
        throw std::runtime_error("cannot load MoE metadata");
    self.experts = self.loaded->hparams.n_expert;
    self.active_experts = self.loaded->hparams.n_expert_used();
    self.slots = slots;
    if (self.experts == 0 || self.experts > 65536 || self.active_experts == 0 ||
        slots < self.active_experts || slots > self.experts)
        throw std::runtime_error("MoE slots must cover top-k and not exceed expert count");
    for (const auto& [name, tensor] : self.loaded->tensors_by_name) {
        const auto id = gguf_find_tensor(metadata.get(), name.c_str());
        if (id < 0 || tensor->view_src || !ggml_is_contiguous(tensor) ||
            tensor->type != gguf_get_tensor_type(metadata.get(), id) ||
            ggml_nbytes(tensor) != gguf_get_tensor_size(metadata.get(), id))
            throw std::runtime_error("unsupported MoE virtual/transformed tensor: " + name);
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] != gguf_get_tensor_ne(metadata.get(), id)[dim])
                throw std::runtime_error("MoE source tensor shape mismatch");
        }
        unsigned projection;
        const int block = expert_block(name, projection);
        const auto original = ggml_nbytes(tensor);
        self.stats.logical_weight_bytes = checked_add(self.stats.logical_weight_bytes, original);
        std::size_t slab = 0, buffer_offset = self.stats.common_bytes;
        if (block >= 0) {
            if (static_cast<std::size_t>(block) >= self.loaded->hparams.n_layer() ||
                self.loaded->hparams.n_expert_used(block) != self.active_experts ||
                tensor->ne[2] != static_cast<int64_t>(self.experts) || tensor->ne[3] != 1 ||
                original % self.experts || tensor->nb[2] != original / self.experts)
                throw std::runtime_error("unsupported MoE expert slab geometry");
            auto& layer = self.layers.try_emplace(block, self.experts, slots).first->second;
            if (layer.projection_mask & projection)
                throw std::runtime_error("duplicate expert FFN projection");
            layer.projection_mask |= projection;
            slab = original / self.experts;
            buffer_offset = layer.bytes;
            layer.bytes = checked_add(layer.bytes, align64(checked_mul(slab, slots)));
            layer.weights.push_back(self.weights.size());
            self.expert_weights.emplace(tensor, self.weights.size());
        } else {
            self.stats.common_bytes = checked_add(self.stats.common_bytes, align64(original));
        }
        self.weights.push_back({tensor,
                                checked_add(gguf_get_data_offset(metadata.get()),
                                            gguf_get_tensor_offset(metadata.get(), id)),
                                original, slab, buffer_offset, block, projection});
    }
    if (self.layers.size() != self.loaded->hparams.n_layer())
        throw std::runtime_error("incomplete MoE decoder layers");
    for (const auto& [block, layer] : self.layers) {
        if (layer.projection_mask != 7)
            throw std::runtime_error("MoE spike requires separate gate/up/down expert weights");
        self.stats.expert_cache_bytes = checked_add(self.stats.expert_cache_bytes, layer.bytes);
    }
    if (checked_add(self.stats.common_bytes, self.stats.expert_cache_bytes) > budget)
        throw std::runtime_error("MoE common weights and expert slots exceed weight budget");
    self.common = allocate_weights(self.stats.common_bytes);
    for (auto& [block, layer] : self.layers)
        layer.storage = allocate_weights(layer.bytes);
    for (auto& weight : self.weights) {
        auto* buffer =
            weight.block < 0 ? self.common.get() : self.layers.at(weight.block).storage.get();
        if (weight.block >= 0) {
            weight.tensor->ne[2] = static_cast<int64_t>(slots);
            weight.tensor->nb[3] = checked_mul(weight.slab_bytes, slots);
        }
        auto* target = static_cast<unsigned char*>(ggml_backend_buffer_get_base(buffer)) +
                       weight.buffer_offset;
        weight.tensor->buffer = nullptr;
        if (ggml_backend_tensor_alloc(buffer, weight.tensor, target) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("cannot attach MoE weight buffer");
        if (weight.block < 0)
            self.read(weight.file_offset, target, weight.original_bytes, false);
    }
    self.loaded->hparams.no_alloc = false;
}

MoeStreamingModel::~MoeStreamingModel() = default;

void MoeStreamingModel::create_context(std::uint32_t n_ctx, std::uint32_t batch, int threads) {
    auto& self = *impl_;
    if (self.context || self.poisoned || self.cancelled.load(std::memory_order_relaxed) || !n_ctx ||
        !batch || threads <= 0)
        throw std::runtime_error("invalid/repeated MoE context creation");
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = batch;
    params.n_ubatch = self.physical_ubatch;
    params.n_threads = params.n_threads_batch = threads;
    params.offload_kqv = params.op_offload = false;
    params.cb_eval = &Impl::evaluate;
    params.cb_eval_user_data = &self;
    self.context.reset(llama_init_from_model(self.loaded.get(), params));
    if (!self.context)
        throw std::runtime_error("cannot create MoE streaming context");
}

void MoeStreamingModel::decode(llama_batch batch) {
    auto& self = *impl_;
    if (!self.context || self.poisoned)
        throw std::runtime_error("MoE context unavailable or poisoned");
    self.has_logits = false;
    try {
        if (self.cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("MoE streaming cancelled");
        if (batch.n_tokens <= 0 || !batch.token || batch.embd ||
            (batch.logits && std::none_of(batch.logits, batch.logits + batch.n_tokens,
                                          [](int8_t value) { return value != 0; })))
            throw std::runtime_error("MoE spike needs a non-empty token batch with logits");
        if (llama_decode(self.context.get(), batch) != 0 || self.active_layer >= 0 ||
            self.pending_ids)
            throw std::runtime_error("incomplete MoE streaming decode");
        if (self.cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("MoE streaming cancelled");
        self.has_logits = true;
        self.snapshot_stats();
    } catch (...) {
        self.restore_ids();
        self.poisoned = true;
        self.snapshot_stats();
        throw;
    }
}

void MoeStreamingModel::cancel() noexcept {
    impl_->cancelled.store(true, std::memory_order_relaxed);
}
const float* MoeStreamingModel::logits() const {
    if (!impl_->context || !impl_->has_logits || impl_->poisoned ||
        impl_->cancelled.load(std::memory_order_relaxed))
        throw std::runtime_error("no valid MoE streaming logits");
    return llama_get_logits(impl_->context.get());
}
const llama_model* MoeStreamingModel::model() const {
    return impl_->loaded.get();
}
const MoeStreamingStats& MoeStreamingModel::stats() const {
    return impl_->stats;
}

} // namespace xllama::experimental
