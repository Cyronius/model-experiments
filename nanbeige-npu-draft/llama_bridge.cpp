// Small C ABI for timing the exact GGUF target with an external PyTorch drafter.
// Build against the local llama.cpp shared library; no model code is copied here.
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

struct bridge {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch_ext * batch = nullptr;
    const llama_vocab * vocab = nullptr;
};

extern "C" {

bridge * bridge_open(const char * path, int32_t gpu_layers, int32_t context, int32_t batch_size, int32_t threads) {
    static bool backend_started = false;
    if (!backend_started) {
        llama_log_set([](ggml_log_level, const char * message, void *) {
            if (std::strstr(message, "offloaded") ||
                std::strstr(message, "Vulkan")) {
                std::fputs(message, stderr);
            }
        }, nullptr);
        llama_backend_init();
        backend_started = true;
    }
    auto * b = new (std::nothrow) bridge();
    if (!b) return nullptr;
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    b->model = llama_model_load_from_file(path, model_params);
    if (!b->model) {
        delete b;
        return nullptr;
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = context;
    context_params.n_batch = batch_size;
    context_params.n_ubatch = batch_size;
    context_params.n_seq_max = 1;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    context_params.embeddings = true;
    context_params.no_perf = true;
    b->ctx = llama_init_from_model(b->model, context_params);
    if (!b->ctx) {
        llama_model_free(b->model);
        delete b;
        return nullptr;
    }
    b->vocab = llama_model_get_vocab(b->model);
    b->batch = llama_batch_ext_init(b->ctx);
    if (!b->batch) {
        llama_free(b->ctx);
        llama_model_free(b->model);
        delete b;
        return nullptr;
    }
    return b;
}

void bridge_close(bridge * b) {
    if (!b) return;
    if (b->batch) llama_batch_ext_free(b->batch);
    if (b->ctx) llama_free(b->ctx);
    if (b->model) llama_model_free(b->model);
    delete b;
}

int32_t bridge_vocab_size(const bridge * b) { return llama_vocab_n_tokens(b->vocab); }
int32_t bridge_hidden_size(const bridge * b) { return llama_model_n_embd_out(b->model); }
int32_t bridge_eos(const bridge * b) { return llama_vocab_eos(b->vocab); }
bool bridge_is_eog(const bridge * b, int32_t token) { return llama_vocab_is_eog(b->vocab, token); }

int32_t bridge_tokenize(const bridge * b, const char * text, int32_t length, int32_t * out, int32_t capacity) {
    return llama_tokenize(b->vocab, text, length, out, capacity, true, true);
}

void bridge_clear(bridge * b) { llama_memory_clear(llama_get_memory(b->ctx), false); }

bool bridge_rollback(bridge * b, int32_t from_position) {
    return llama_memory_seq_rm(llama_get_memory(b->ctx), 0, from_position, -1);
}

// If all_outputs is false, only the final token's logits and hidden state are
// retained. If true, every token is retained for MTP prefill or verification.
int32_t bridge_decode(bridge * b, const int32_t * tokens, int32_t count,
                      int32_t start_position, bool all_outputs) {
    if (count < 1) return -1;
    llama_batch_ext_clear(b->batch);
    for (int32_t i = 0; i < count; ++i) {
        const int32_t index = llama_batch_ext_add_token(b->batch, 0, tokens[i]);
        if (index < 0) return -2;
        const llama_pos pos = start_position + i;
        if (!llama_batch_ext_set_pos(b->batch, index, &pos)) return -3;
        if (all_outputs || i == count - 1) {
            if (!llama_batch_ext_set_output_logits(b->batch, index, true)) return -4;
        }
    }
    return llama_process(b->ctx, LLAMA_PROCESS_TYPE_DECODE, b->batch);
}

const float * bridge_logits(bridge * b, int32_t index) { return llama_get_logits_ith(b->ctx, index); }
const float * bridge_hidden(bridge * b, int32_t index) { return llama_get_embeddings_ith(b->ctx, index); }

}
