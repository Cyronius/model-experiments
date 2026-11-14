// MemDraft: a drafter that only looks things up.
//
// It proposes the next few tokens from two places and runs no model:
//   - a table built ahead of time (ngram_study.py export): for the last 1..K tokens, the token the
//     target model most often chose next, and how often that guess was right on held-out text;
//   - the current request: the token that followed the same last 1..K tokens earlier in it.
// Each proposal carries a confidence in 255ths. The most confident one is drafted, and the chain
// goes on while the chance that the whole chain is accepted stays at or above `tau`. With `pad`
// every draft is exactly n_max tokens, which keeps the target's batch shape the same at every step.
//
// Nothing here depends on llama.cpp. ngram_study.Exported is the same logic in numpy, and
// replay.cpp checks the two against each other.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace memdraft {

constexpr uint32_t K_MAX      = 8;    // longest context a table may use
constexpr uint32_t BUCKET     = 8;    // slots per bucket: one cache line
constexpr uint32_t LABEL_BITS = 18;   // low hash bits ngram_study drops when it counts

// One more token folded into a context hash (splitmix64 finalizer); ngram_study.mix.
static inline uint64_t mix(uint64_t h, uint64_t t) {
    uint64_t x = h + (t + 1) * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

// The table file: a 64-byte header, the confidences, then the slots.
// A slot is tag (32 bits) | token (24 bits) | cell (8 bits); 0 means empty.
//   version 1: the cell is the confidence in 255ths; in-request confidences in 255ths.
//   version 2: the cell is a count bucket and a ratio bucket (ngram_study.static_cell), and the
//              file holds each order's confidence per cell, and the in-request ones, as floats.
//              The reranker reads the buckets; the drafter alone reads only the confidences.
struct table {
    uint32_t version   = 0;
    uint32_t K         = 0;   // longest context, in tokens
    uint32_t n_seen    = 0;   // in-request confidences per order (earlier occurrences 0..n_seen-1)
    uint32_t n_cells   = 0;   // version 2: cells per order
    uint32_t n_ratio   = 0;   // version 2: ratio buckets per count bucket
    uint64_t n_buckets = 0;   // a power of two
    uint64_t n_entries = 0;

    std::vector<float>    seen_f;     // [K][n_seen]
    std::vector<float>    cell_f;     // [K][n_cells]; [K][256] for version 1
    std::vector<uint64_t> slots;      // [n_buckets][BUCKET]

    bool load(const std::string & path, std::string & err) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        struct {
            char     magic[8];
            uint32_t version, K, n_seen, bucket;
            uint64_t n_buckets, n_entries;
            uint32_t n_cells, n_ratio;
            uint8_t  pad[16];
        } head;
        static_assert(sizeof(head) == 64, "table header is 64 bytes");
        bool ok = std::fread(&head, sizeof(head), 1, f) == 1;
        ok = ok && std::memcmp(head.magic, "MDTABLE1", 8) == 0 && (head.version == 1 || head.version == 2);
        ok = ok && head.bucket == BUCKET && head.K >= 1 && head.K <= K_MAX && head.n_seen >= 1;
        ok = ok && head.n_buckets > 0 && (head.n_buckets & (head.n_buckets - 1)) == 0;
        ok = ok && (head.version == 1 || (head.n_cells >= 1 && head.n_cells <= 256 && head.n_ratio >= 1 &&
                                          head.n_cells % head.n_ratio == 0));
        if (ok) {
            version   = head.version;
            K         = head.K;
            n_seen    = head.n_seen;
            n_buckets = head.n_buckets;
            n_entries = head.n_entries;
            const size_t n_ctx = (size_t) K * n_seen;
            if (version == 1) {
                n_cells = 256;
                n_ratio = 0;
                std::vector<uint8_t> q((n_ctx + 7) / 8 * 8);
                ok = std::fread(q.data(), 1, q.size(), f) == q.size();
                seen_f.resize(n_ctx);
                for (size_t i = 0; i < n_ctx; ++i) {
                    seen_f[i] = (float) q[i] / 255.0f;
                }
                cell_f.resize((size_t) K * 256);
                for (size_t i = 0; i < cell_f.size(); ++i) {
                    cell_f[i] = (float) (i % 256) / 255.0f;
                }
            } else {
                n_cells = head.n_cells;
                n_ratio = head.n_ratio;
                seen_f.resize(n_ctx);
                cell_f.resize((size_t) K * n_cells);
                ok = std::fread(seen_f.data(), sizeof(float), n_ctx, f) == n_ctx;
                ok = ok && std::fread(cell_f.data(), sizeof(float), cell_f.size(), f) == cell_f.size();
                const size_t used = (n_ctx + cell_f.size()) * sizeof(float);
                char         skip[8];
                ok = ok && std::fread(skip, 1, (8 - used % 8) % 8, f) == (8 - used % 8) % 8;
            }
            slots.resize((size_t) n_buckets * BUCKET);
            ok = ok && std::fread(slots.data(), sizeof(uint64_t), slots.size(), f) == slots.size();
        }
        std::fclose(f);
        if (!ok) {
            err = path + " is not a MemDraft table this build can read";
        }
        return ok;
    }

    // The confidence of a table entry at order k, from the low byte of its slot.
    inline float static_conf(uint32_t k, uint32_t cell) const {
        return cell_f[(size_t) (k - 1) * n_cells + (cell < n_cells ? cell : n_cells - 1)];
    }

    // The slot for an order-k context hash, or 0.
    inline uint64_t find(uint32_t k, uint64_t h) const {
        const uint64_t   g   = mix(h >> LABEL_BITS, k);
        const uint64_t * b   = &slots[(g & (n_buckets - 1)) * BUCKET];
        const uint64_t   tag = g >> 32;
        for (uint32_t i = 0; i < BUCKET; ++i) {
            if (b[i] != 0 && (b[i] >> 32) == tag) {
                return b[i];
            }
        }
        return 0;
    }

    inline float seen_conf(uint32_t k, uint32_t seen) const {
        return seen_f[(size_t) (k - 1) * n_seen + (seen < n_seen ? seen : n_seen - 1)];
    }
};

// What the lookups propose at one position: slot 2*(k-1) is the table at order k, 2*(k-1)+1 the
// current request. The reranker reads all of them; the drafter alone takes the most confident.
struct proposals {
    static constexpr uint32_t S_MAX = 2 * K_MAX;

    uint32_t S = 0;
    bool     H[S_MAX];    // proposes something
    int32_t  T[S_MAX];    // the token, or -1
    float    CF[S_MAX];   // its confidence
    float    A[S_MAX];    // table: count bucket; request: earlier occurrences (both scaled to 0..1)
    float    B[S_MAX];    // table: ratio bucket (scaled); request: 0
};

// y += a * w over n floats: the reranker's only arithmetic. llama.cpp builds common/ without
// -march, so on x86-64 an AVX2 version is chosen at load time when the CPU has it (9x faster here).
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
__attribute__((target_clones("avx2", "default")))
#endif
static void axpy(float * __restrict y, const float * __restrict w, float a, uint32_t n) {
    for (uint32_t o = 0; o < n; ++o) {
        y[o] += a * w[o];
    }
}

// A small MLP that scores each distinct proposed token from the proposals, the kinds of the token
// and the two before it, how common they are, and projections of the target's embeddings (the
// candidate's output embedding, the input embeddings of the six tokens before it). The best score
// is drafted. Trained in rerank-scratch/rerank7.py; written by export_reranker.py. Its features
// must stay exactly those of rerank.build.
struct reranker {
    uint32_t V = 0, S = 0, n_cls = 0, n_ctx = 0, d = 0, h1 = 0, h2 = 0, h3 = 0;
    uint32_t n_x = 0;     // proposal features (8*S + 5 + 3*n_cls + 4)

    std::vector<uint8_t> cls;       // [V]
    std::vector<float>   logfreq;   // [V]
    std::vector<float>   p_out;     // [V][d]: output embedding, projected
    std::vector<float>   p_in;      // [V][d]: input embedding, projected
    std::vector<float>   w1;        // [n_x + d + n_ctx*d][h1], input-major
    std::vector<float>   b1, w2, b2, w3, b3, w4;   // w2 [h1][h2], w3 [h2][h3], w4 [h3]
    float                b4 = 0.0f;

    bool load(const std::string & path, std::string & err) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        struct {
            char     magic[8];
            uint32_t version, V, S, n_cls, n_ctx, d, h1, h2, h3, n_x;
            uint8_t  pad[16];
        } head;
        static_assert(sizeof(head) == 64, "reranker header is 64 bytes");
        bool ok = std::fread(&head, sizeof(head), 1, f) == 1;
        ok = ok && std::memcmp(head.magic, "MDRERANK", 8) == 0 && head.version == 1;
        ok = ok && head.S >= 2 && head.S <= proposals::S_MAX && head.n_ctx >= 2 && head.n_ctx <= K_MAX;
        ok = ok && head.n_x == 8 * head.S + 5 + 3 * head.n_cls + 4 && head.V > 0 && head.d > 0;
        auto read = [&](auto & v, size_t n) {
            v.resize(n);
            ok = ok && std::fread(v.data(), sizeof(v[0]), n, f) == n;
        };
        if (ok) {
            V = head.V, S = head.S, n_cls = head.n_cls, n_ctx = head.n_ctx, d = head.d;
            h1 = head.h1, h2 = head.h2, h3 = head.h3, n_x = head.n_x;
            read(cls, V);
            char skip[8];
            ok = ok && std::fread(skip, 1, (8 - V % 8) % 8, f) == (8 - V % 8) % 8;
            read(logfreq, V);
            read(p_out, (size_t) V * d);
            read(p_in, (size_t) V * d);
            read(w1, (size_t) (n_x + d + n_ctx * d) * h1);
            read(b1, h1);
            read(w2, (size_t) h1 * h2);
            read(b2, h2);
            read(w3, (size_t) h2 * h3);
            read(b3, h3);
            read(w4, h3);
            std::vector<float> last;
            read(last, 1);
            b4 = ok ? last[0] : 0.0f;
            for (uint8_t c : cls) {
                ok = ok && c < n_cls;
            }
        }
        std::fclose(f);
        if (!ok) {
            err = path + " is not a MemDraft reranker this build can read";
        }
        return ok;
    }

    size_t bytes() const {
        return cls.size() + (logfreq.size() + p_out.size() + p_in.size() + w1.size() + w2.size() + w3.size()) * sizeof(float);
    }

    // The best proposed token at a position whose last tokens are ctx[0..n_ctx) (most recent first),
    // and its score as a probability; -1 when nothing is proposed. Ties go to the later slot.
    int32_t pick(const proposals & p, const int32_t * ctx, float & prob) const {
        uint32_t S_ = p.S < S ? p.S : S;
        float    top  = 0.0f;
        int      base = -1;
        for (uint32_t s = 0; s < S_; ++s) {
            top = p.CF[s] > top ? p.CF[s] : top;
        }
        if (top >= 1e-9f) {
            for (int s = (int) S_ - 1; s >= 0; --s) {
                if (p.CF[s] >= top) {
                    base = s;
                    break;
                }
            }
        }

        bool  ctx_done = false;
        float best     = 0.0f;
        int32_t best_tok = -1;
        for (uint32_t s = 0; s < S_; ++s) {
            if (!p.H[s]) {
                continue;
            }
            bool earlier = false;
            for (uint32_t r = 0; r < s && !earlier; ++r) {
                earlier = p.H[r] && p.T[r] == p.T[s];
            }
            if (earlier) {
                continue;
            }
            if (!ctx_done) {
                ctx_part(ctx);
                ctx_done = true;
            }
            const float z = score(p, s, base, ctx);
            if (best_tok < 0 || z >= best) {
                best     = z;
                best_tok = p.T[s];
            }
        }
        prob = best_tok < 0 ? 0.0f : 1.0f / (1.0f + std::exp(-best));
        return best_tok;
    }

  private:
    mutable std::vector<float> x_, a1_, a2_, a3_, cp_;

    static void axpy(float * y, const float * w, float a, uint32_t n) {
        memdraft::axpy(y, w, a, n);
    }

    // The first layer's share from the tokens before the position, the same for every candidate.
    void ctx_part(const int32_t * ctx) const {
        cp_.assign(b1.begin(), b1.end());
        for (uint32_t c = 0; c < n_ctx; ++c) {
            const uint32_t t = ctx[c] >= 0 && (uint32_t) ctx[c] < V ? (uint32_t) ctx[c] : 0;
            const float *  e = &p_in[(size_t) t * d];
            const float *  w = &w1[(size_t) (n_x + d + c * d) * h1];
            for (uint32_t j = 0; j < d; ++j) {
                axpy(cp_.data(), w + (size_t) j * h1, e[j], h1);
            }
        }
    }

    float score(const proposals & p, uint32_t s, int base, const int32_t * ctx) const {
        const int32_t tok = p.T[s];
        const uint32_t S_ = S;
        x_.assign(n_x, 0.0f);
        float * X = x_.data();
        float own = 0.0f, rival = 0.0f, n_ag = 0.0f;
        for (uint32_t r = 0; r < S_; ++r) {
            const bool  ag = r < p.S && p.H[r] && p.T[r] == tok;
            const float cf = r < p.S ? p.CF[r] : 0.0f;
            if (ag) {
                X[r]          = 1.0f;
                X[S_ + r]     = cf;
                X[2 * S_ + r] = p.A[r];
                X[3 * S_ + r] = p.B[r];
                own           = cf > own ? cf : own;
                n_ag += 1.0f;
            } else {
                rival = cf > rival ? cf : rival;
            }
            if (r < p.S) {
                X[4 * S_ + r] = p.H[r] ? 1.0f : 0.0f;
                X[5 * S_ + r] = cf;
                X[6 * S_ + r] = p.A[r];
                X[7 * S_ + r] = p.B[r];
            }
        }
        float * v = X + 8 * S_;
        v[0] = own;
        v[1] = rival;
        v[2] = n_ag / (float) S_;
        v[3] = own - rival;
        v[4] = base >= 0 && p.H[base] && p.T[base] == tok ? 1.0f : 0.0f;
        const uint32_t t  = (uint32_t) tok < V ? (uint32_t) tok : 0;
        const uint32_t p0 = ctx[0] >= 0 && (uint32_t) ctx[0] < V ? (uint32_t) ctx[0] : 0;
        const uint32_t p1 = ctx[1] >= 0 && (uint32_t) ctx[1] < V ? (uint32_t) ctx[1] : 0;
        float * c = v + 5;
        c[cls[t]]             = 1.0f;
        c[n_cls + cls[p0]]    = 1.0f;
        c[2 * n_cls + cls[p1]] = 1.0f;
        float * q = c + 3 * n_cls;
        q[0] = logfreq[t];
        q[1] = logfreq[p0];
        q[2] = tok == ctx[0] ? 1.0f : 0.0f;
        q[3] = tok == ctx[1] ? 1.0f : 0.0f;

        a1_.assign(cp_.begin(), cp_.end());
        for (uint32_t i = 0; i < n_x; ++i) {
            if (X[i] != 0.0f) {
                axpy(a1_.data(), &w1[(size_t) i * h1], X[i], h1);
            }
        }
        const float * e = &p_out[(size_t) t * d];
        for (uint32_t j = 0; j < d; ++j) {
            axpy(a1_.data(), &w1[(size_t) (n_x + j) * h1], e[j], h1);
        }
        a2_.assign(b2.begin(), b2.end());
        for (uint32_t i = 0; i < h1; ++i) {
            if (a1_[i] > 0.0f) {
                axpy(a2_.data(), &w2[(size_t) i * h2], a1_[i], h2);
            }
        }
        a3_.assign(b3.begin(), b3.end());
        for (uint32_t i = 0; i < h2; ++i) {
            if (a2_[i] > 0.0f) {
                axpy(a3_.data(), &w3[(size_t) i * h3], a2_[i], h3);
            }
        }
        float z = b4;
        for (uint32_t i = 0; i < h3; ++i) {
            if (a3_[i] > 0.0f) {
                z += a3_[i] * w4[i];
            }
        }
        return z;
    }
};

// What followed each context earlier in the current request, and how many times it was seen.
struct request_index {
    struct entry {
        uint64_t key;     // 0 means empty
        int32_t  token;   // what followed the latest occurrence
        uint32_t seen;
    };

    std::vector<entry> slots;
    size_t             used      = 0;
    size_t             n_indexed = 0;   // tokens[0..n_indexed) are in the index

    static inline uint64_t key_of(uint32_t k, uint64_t h) {
        const uint64_t g = mix(h, 256 + k);
        return g ? g : 1;
    }

    void clear() {
        slots.assign(1024, entry{0, 0, 0});
        used      = 0;
        n_indexed = 0;
    }

    entry * probe(std::vector<entry> & in, uint64_t key) const {
        const size_t mask = in.size() - 1;
        size_t       i    = key & mask;
        while (in[i].key != 0 && in[i].key != key) {
            i = (i + 1) & mask;
        }
        return &in[i];
    }

    const entry * find(uint64_t key) const {
        const size_t mask = slots.size() - 1;
        size_t       i    = key & mask;
        while (slots[i].key != 0) {
            if (slots[i].key == key) {
                return &slots[i];
            }
            i = (i + 1) & mask;
        }
        return nullptr;
    }

    // Record that tokens[j] followed the 1..K tokens before it.
    void add(const int32_t * tokens, size_t j, uint32_t K) {
        if ((used + K) * 2 > slots.size()) {
            std::vector<entry> bigger(slots.size() * 2, entry{0, 0, 0});
            for (const entry & e : slots) {
                if (e.key != 0) {
                    *probe(bigger, e.key) = e;
                }
            }
            slots.swap(bigger);
        }
        uint64_t h = 0;
        for (uint32_t k = 1; k <= K && k <= j; ++k) {
            h         = mix(h, (uint32_t) tokens[j - k]);
            entry * e = probe(slots, key_of(k, h));
            if (e->key == 0) {
                e->key  = key_of(k, h);
                e->seen = 0;
                used++;
            }
            e->token = tokens[j];
            e->seen++;
        }
    }
};

struct drafter {
    const table * tab         = nullptr;
    bool          use_static  = true;
    bool          use_context = true;
    float         tau         = 0.3f;   // stop when the chain's chance of full acceptance drops below this
    int           n_max       = 4;
    bool          pad         = false;  // always draft n_max tokens: past tau, keep going with the best guess
                                        // (the last token when there is none), so every step has one batch shape
    const reranker * rr       = nullptr;  // pick among the proposals with this (needs a version 2 table);
                                          // its score stands in for the confidence

    request_index index;
    proposals     prop;

    // Why this drafter cannot run with `r`, or "" when it can.
    std::string check(const reranker & r) const {
        if (tab->version < 2) {
            return "the reranker needs a version 2 table (export_reranker.py)";
        }
        if (r.S != 2 * tab->K || r.n_ctx > tab->K) {
            return "the reranker was trained for another context length than this table's";
        }
        return "";
    }

    void begin() {
        index.clear();
    }

    // tokens[0..n) is everything committed so far (prompt, accepted tokens, the last sampled one).
    // Appends the draft to `out`.
    void draft(const int32_t * tokens, size_t n, std::vector<int32_t> & out) {
        if (index.slots.empty() || n < index.n_indexed) {
            index.clear();   // a new or rewritten context: index it again from the start
        }
        const uint32_t K = tab->K;
        if (use_context) {
            for (size_t j = index.n_indexed > 0 ? index.n_indexed : 1; j < n; ++j) {
                index.add(tokens, j, K);
            }
            index.n_indexed = n;
        }

        int32_t  ctx[K_MAX] = {0};
        uint32_t avail = n < K ? (uint32_t) n : K;
        for (uint32_t i = 0; i < avail; ++i) {
            ctx[i] = tokens[n - 1 - i];
        }

        float running = 1.0f;
        for (int d = 0; d < n_max; ++d) {
            uint64_t h        = 0;
            float    best     = 0.0f;
            int32_t  best_tok = -1;
            if (rr) {
                prop.S = 2 * K;
                for (uint32_t s = 0; s < prop.S; ++s) {
                    prop.H[s] = false, prop.T[s] = -1, prop.CF[s] = prop.A[s] = prop.B[s] = 0.0f;
                }
            }
            for (uint32_t k = 1; k <= avail; ++k) {
                h = mix(h, (uint32_t) ctx[k - 1]);
                if (use_static) {
                    const uint64_t s = tab->find(k, h);
                    const float    c = s ? tab->static_conf(k, (uint32_t) (s & 0xFF)) : 0.0f;
                    if (c > 0.0f && c >= best) {   // ties go to the longer context
                        best     = c;
                        best_tok = (int32_t) ((s >> 8) & 0xFFFFFF);
                    }
                    if (rr && c > 0.0f) {
                        const uint32_t i = 2 * (k - 1), cell = (uint32_t) (s & 0xFF);
                        prop.H[i]  = true;
                        prop.T[i]  = (int32_t) ((s >> 8) & 0xFFFFFF);
                        prop.CF[i] = c;
                        prop.A[i]  = (float) ((double) (cell / tab->n_ratio + 1) / (tab->n_cells / tab->n_ratio));
                        prop.B[i]  = (float) ((double) (cell % tab->n_ratio + 1) / tab->n_ratio);
                    }
                }
                if (use_context) {
                    const request_index::entry * e = index.find(request_index::key_of(k, h));
                    const float                  c = e ? tab->seen_conf(k, e->seen) : 0.0f;
                    if (c > 0.0f && c >= best) {
                        best     = c;
                        best_tok = e->token;
                    }
                    if (rr && c > 0.0f) {
                        const uint32_t i = 2 * (k - 1) + 1, cap = tab->n_seen - 1;
                        prop.H[i]  = true;
                        prop.T[i]  = e->token;
                        prop.CF[i] = c;
                        prop.A[i]  = (float) ((double) (e->seen < cap ? e->seen : cap) / cap);
                    }
                }
            }
            if (rr) {
                best_tok = rr->pick(prop, ctx, best);
            }
            running *= best;
            if (best_tok < 0 || running < tau) {
                if (!pad) {
                    break;
                }
                if (best_tok < 0) {
                    best_tok = ctx[0];
                }
            }
            out.push_back(best_tok);
            for (uint32_t i = K - 1; i > 0; --i) {
                ctx[i] = ctx[i - 1];
            }
            ctx[0] = best_tok;
            avail  = avail < K ? avail + 1 : K;
        }
    }
};

} // namespace memdraft
