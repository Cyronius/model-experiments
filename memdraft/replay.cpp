// Decode recorded requests with the MemDraft drafter and count what the target would accept.
//
// No model runs. The stream (ngram_study.write_stream) holds, per request, the tokens the model
// wrote and its top choice at each answer position. Each step drafts from the tokens so far,
// keeps the leading draft tokens that match, adds the target's own next token, and moves on.
// The counts must equal ngram_study's for the same table; `build_table.py --check` compares them.
//
//   g++ -O2 -std=c++17 -o replay replay.cpp
//   ./replay TABLE STREAM --mode greedy|sampled --policy static|context|both --tau 0.3 --n-max 8 --pad 0
//            [--reranker FILE.mdr] [--dump FILE]
//
// --dump also drafts at every position that has a top choice, as if all before it were committed,
// and writes those drafts (n_max int32 each, -2 past the end of a draft) for export_reranker.py to
// compare with the numpy reranker.
#include "memdraft.h"

#include <chrono>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s TABLE STREAM [--mode greedy|sampled] [--policy static|context|both] [--tau F] [--n-max N] [--pad 0|1]\n", argv[0]);
        return 2;
    }
    std::string mode = "greedy", policy = "both", rerank_path, dump_path;
    float       tau   = 0.3f;
    int         n_max = 8;
    bool        pad   = false;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--mode") {
            mode = argv[i + 1];
        } else if (flag == "--policy") {
            policy = argv[i + 1];
        } else if (flag == "--tau") {
            tau = std::strtof(argv[i + 1], nullptr);
        } else if (flag == "--n-max") {
            n_max = std::atoi(argv[i + 1]);
        } else if (flag == "--pad") {
            pad = std::atoi(argv[i + 1]) != 0;
        } else if (flag == "--reranker") {
            rerank_path = argv[i + 1];
        } else if (flag == "--dump") {
            dump_path = argv[i + 1];
        } else {
            std::fprintf(stderr, "unknown option %s\n", flag.c_str());
            return 2;
        }
    }
    const bool greedy = mode == "greedy";

    memdraft::table tab;
    std::string     err;
    if (!tab.load(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    memdraft::drafter drafter;
    drafter.tab         = &tab;
    drafter.use_static  = policy != "context";
    drafter.use_context = policy != "static";
    drafter.tau         = tau;
    drafter.n_max       = n_max;
    drafter.pad         = pad;
    memdraft::reranker rr;
    if (!rerank_path.empty()) {
        if (!rr.load(rerank_path, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        if (!(err = drafter.check(rr)).empty()) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        drafter.rr = &rr;
    }
    FILE * dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");

    FILE * f = std::fopen(argv[2], "rb");
    struct {
        char     magic[8];
        uint32_t n_seq, pad;
    } head;
    if (!f || std::fread(&head, sizeof(head), 1, f) != 1 || std::memcmp(head.magic, "MDSTREAM", 8) != 0) {
        std::fprintf(stderr, "%s is not a MemDraft stream\n", argv[2]);
        return 1;
    }

    long long            tokens = 0, steps = 0, drafted = 0, accepted = 0, rejected = 0, reshaped = 0;
    double               seconds = 0;
    std::vector<int32_t> tok, top, chain;
    for (uint32_t s = 0; s < head.n_seq; ++s) {
        uint32_t size[2];
        if (std::fread(size, sizeof(uint32_t), 2, f) != 2) {
            std::fprintf(stderr, "stream ends early at request %u\n", s);
            return 1;
        }
        const long long n_prompt = size[0], n_answer = size[1], end = n_prompt + n_answer;
        tok.resize(end);
        top.resize(n_answer);
        if (std::fread(tok.data(), sizeof(int32_t), end, f) != (size_t) end ||
            std::fread(top.data(), sizeof(int32_t), n_answer, f) != (size_t) n_answer) {
            std::fprintf(stderr, "stream ends early at request %u\n", s);
            return 1;
        }
        // the positions with a stored top choice: first..last
        long long first = -1, last = -1;
        for (long long i = 0; i < n_answer; ++i) {
            if (top[i] >= 0) {
                if (first < 0) {
                    first = n_prompt + i;
                }
                last = n_prompt + i;
            }
        }
        if (first < 0) {
            continue;
        }
        tokens += last - first + 1;
        if (dump) {
            drafter.begin();
            for (long long j = first; j <= last; ++j) {
                if (top[j - n_prompt] < 0) {
                    continue;
                }
                chain.clear();
                drafter.draft(tok.data(), (size_t) j, chain);
                chain.resize(n_max, -2);
                std::fwrite(chain.data(), sizeof(int32_t), n_max, dump);
            }
        }
        drafter.begin();
        long long shape = 0;   // draft length of the previous step; the prompt counts as a different shape
        for (long long j = first; j <= last;) {
            chain.clear();
            const auto t0 = std::chrono::steady_clock::now();
            drafter.draft(tok.data(), (size_t) j, chain);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            long long kept = 0, right = 0;
            bool      still_right = true;
            for (size_t d = 0; d < chain.size(); ++d) {
                const long long at = j + (long long) d;
                if (at >= end) {
                    break;
                }
                const int32_t truth = greedy ? top[at - n_prompt] : tok[at];
                if (truth < 0) {
                    break;
                }
                kept++;
                if (still_right && chain[d] == truth) {
                    right++;
                } else {
                    still_right = false;
                }
                // the text was sampled: past a position where it left the top choice, greedy
                // decoding would have gone somewhere the stream does not record
                if (greedy && tok[at] != top[at - n_prompt]) {
                    break;
                }
            }
            steps++;
            drafted += kept;
            accepted += right;
            rejected += right < kept;
            reshaped += j == first || kept != shape;
            shape = kept;
            j += right + 1;
        }
    }
    std::fclose(f);
    if (dump) {
        std::fclose(dump);
    }
    std::printf("{\"mode\": \"%s\", \"policy\": \"%s\", \"tau\": %g, \"n_max\": %d, \"tokens\": %lld, \"steps\": %lld, "
                "\"drafted\": %lld, \"accepted\": %lld, \"rejected\": %lld, \"reshaped\": %lld, \"draft_ns_per_step\": %.1f}\n",
                mode.c_str(), policy.c_str(), tau, n_max, tokens, steps, drafted, accepted, rejected, reshaped,
                steps ? seconds * 1e9 / steps : 0.0);
    return 0;
}
