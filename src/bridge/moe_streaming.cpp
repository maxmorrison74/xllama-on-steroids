// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
#include "xllama/moe_streaming.h"

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/stat.h>
#endif

namespace xllama {
namespace {

std::string now_utc() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);
    return buf;
}

bool write_csv(const std::string& path, const InferenceResult& res,
               const MoeStreamingParams& params) {
#ifdef _WIN32
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"w");
#else
    FILE* fp = std::fopen(path.c_str(), "w");
#endif
    if (!fp)
        return false;
    std::fputs("model_name,prompt,n_predict,n_ctx,n_threads,n_gpu_layers,"
               "success,output_text,t_p_eval_ms,t_gen_ms,tokens_generated,error_msg,host,date\n",
               fp);
    // Escape commas/newlines in output and error message minimally.
    auto esc = [](const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            if (c == ',' || c == '"' || c == '\n' || c == '\r')
                out.push_back('"');
            else
                out.push_back(c);
        }
        return out;
    };
    std::fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,\"%s\",%.2f,%.2f,%d,\"%s\",\"xbox-series-x\",\"%s\"\n",
                 params.model_name.c_str(), esc(params.prompt).c_str(), params.n_predict,
                 params.n_ctx, params.n_threads, params.n_gpu_layers, res.success ? 1 : 0,
                 esc(res.output_text).c_str(), res.t_p_eval_ms, res.t_eval_ms, res.n_eval,
                 esc(res.error_msg).c_str(), now_utc().c_str());
    std::fclose(fp);
    return true;
}

} // namespace

bool run_moe_streaming(const MoeStreamingParams& params) {
    InferenceParams ip;
    ip.model_path = resolve_model_path(params.model_name);
    ip.prompt = params.prompt;
    ip.n_predict = params.n_predict;
    ip.n_ctx = params.n_ctx;
    ip.n_threads = params.n_threads;
    ip.n_gpu_layers = params.n_gpu_layers;
    ip.greedy = params.greedy;
    ip.chat_template = false;
    ip.run_membw = false;
    ip.run_diskbw = false;
    ip.run_gpubw = false;
    ip.run_gpugemv = false;

    char lb[256];
    std::snprintf(lb, sizeof(lb), "[xllama] moestream: loading model %s -> %s\n",
                  params.model_name.c_str(), ip.model_path.c_str());
    log_output(lb);

    InferenceResult res = run_inference(ip);

    std::snprintf(lb, sizeof(lb),
                  "[xllama] moestream: %s output_text_len=%zu t_p_eval=%.1f ms "
                  "t_gen=%.1f ms tokens=%d\n",
                  res.success ? "ok" : "fail", res.output_text.size(), res.t_p_eval_ms,
                  res.t_eval_ms, res.n_eval);
    log_output(lb);
    if (!res.error_msg.empty())
        log_output(("[xllama] moestream error: " + res.error_msg + "\n").c_str());

    const std::string csv = resolve_local_path("moestream-result.csv");
    const bool wrote = write_csv(csv, res, params);

#ifdef _WIN32
    FILE* done =
        _wfopen(utf8_to_wstring(resolve_local_path("moestream-result.csv.done")).c_str(), L"w");
#else
    FILE* done = std::fopen(resolve_local_path("moestream-result.csv.done").c_str(), "w");
#endif
    if (done) {
        std::fputs(res.success ? "ok\n" : "fail\n", done);
        std::fclose(done);
    }

    if (wrote)
        log_output("[xllama] moestream-result.csv written\n");
    else
        log_output("[xllama] moestream: cannot write result CSV\n");

    return res.success;
}

} // namespace xllama
