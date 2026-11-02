#SSD - streamed inference — assessment(2026 - 08)

> Claim under test : "SSDs can be used for inference." Verdict up front : **true,
    > but only as weight / KV streaming bounded by file - read bandwidth — never as >
        compute inside the drive — and on Series S it buys capacity,
    not speed.** > The one number this repo can add is the sandboxed NVMe read rate(§4).

                   ##1. What has actually been demonstrated

        | Work | Mechanism | Hardware | Measured | Source | |
        -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --|
        -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -|
        -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -|
        -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -| -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --|
        | Flash - MoE(@anemll, 2026 - 03) | MoE expert streaming from NVMe,
    2–4 - bit experts, hot cache; active experts cut 10→4           | iPhone 17 Pro                                                           | Qwen3.5-397B-A17B, **0.6 tok/s**, 5.5 GB resident               | [Willison](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/), [wccftech](https://wccftech.com/iphone-17-pro-successfully-runs-400b-llm-locally/)                                                            |
| Flash-MoE, same stack                                                                          | as above                                                                                      | M3 Max 48 GB (~17.5 GB/s SSD)                                           | same model (209 GB on disk), **4.4–5.5 tok/s**                  | [Willison](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/)                                                                                                                                                |
| Apple "LLM in a flash" (2023-12)                                                               | dense-model analogue: ReLU sparsity → load ~2–5% of FFN rows; windowing + row-column bundling | OPT-6.7B / Falcon-7B class                                              | models up to ~2× DRAM; 4–5× vs naive paging (CPU), 20–25× (GPU) | [arXiv 2312.11514](https://arxiv.org/pdf/2312.11514)                                                                                                                                                             |
| llama.cpp mmap (merged, default)                                                               | page weights from disk on demand                                                              | community: DeepSeek-R1 671B on 96 GB RAM + 24 GB VRAM, overflow on NVMe | **~1–2 tok/s**; 2–5 GB/s NVMe read observed during decode       | [HF discussion](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/13)                                                                                                                                  |
| llama.cpp [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) (**open, not merged**) | explicit per-layer expert cache + I/O thread pool (`--moe-stream-cache`, O_DIRECT), mmap off  | ~754B GLM MoE                                                           | **~1.8–2.2 tok/s** decode (2.4× vs mmap+cpu-moe)                | PR thread                                                                                                                                                                                                        |
| FlexGen / ZeRO-Inference (2023)                                                                | layer-by-layer weight fetch amortized over huge batches                                       | OPT-175B, 16 GB T4 + SSD                                                | ~1 tok/s **aggregate at batch 144**; interactive latency dismal | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)                                                                                                                                                             |
| Phison aiDAPTIV+ (2026-01 PC push)                                                             | middleware swaps weights/KV to proprietary SSDs; "10× TTFT" = KV-cache persistence            | gpt-oss-120B on 32 GB laptop (claim)                                    | **no tok/s published**; no independent benchmarks               | [Blocks&Files](https://www.blocksandfiles.com/ai-ml/2026/01/07/phison-extends-aidaptiv-to-boost-ai-inference-on-pcs/4090377)                                                                                     |
| Computational storage (SmartSSD, KVNAND, InstInfer, SK hynix HBF)                              | compute in/near the drive                                                                     | research prototypes / roadmap                                           | nothing purchasable runs a transformer in-drive today           | [InstInfer](https://arxiv.org/pdf/2409.04992), [KVNAND](https://arxiv.org/pdf/2512.03608), [HBF](https://www.blocksandfiles.com/flash/2026/02/16/sk-hynix-proposes-hbm-and-hbf-hybrid-for-llm-inference/4091326) |

Pattern: every real demo is **offload + streaming**; the drive never computes.
And every real number is ~0.5–5 tok/s — proofs of capacity, not of speed.

## 2. The physics

Decode is a bandwidth-bound M=1 GEMV: on a purely storage-bound path every
token streams every **active** weight byte once, so
`tok/s ≈ effective read GB/s ÷ active GB` — an upper bound over the bytes
actually fetched from storage (a hot cache shrinks the denominator to the
missed bytes, §2 escapes). The hierarchy this repo has already measured
(Series S):

| Tier                         | Read bandwidth                   | Where measured                                      |
| ---------------------------- | -------------------------------- | --------------------------------------------------- |
| CPU DRAM                     | 12.35 GB/s @1t, ~30 GB/s ceiling | `membw` ([benchmarks.md](benchmarks.md))            |
| GPU pool                     | 119.07 GB/s                      | `gpubw` W3 ([phase15-re-opt.md](phase15-re-opt.md)) |
| NVMe (PCIe 4.0 x2, raw spec) | ~2.4 GB/s                        | vendor spec; sandboxed rate in §4                   |

Dense streaming is therefore dead on arrival for chat: a 4 GB Q4 model at even
the full raw 2.4 GB/s is **~0.6 tok/s** before any compute. The only escapes:

- **MoE sparsity + hot expert cache** — only cache-_misses_ touch the SSD, so
  the denominator shrinks from "model size" to "missed expert bytes/token".
- **KV/prompt-cache on disk** — not decode at all: it removes prefill recompute
  (this is the honest content of Phison's "10× TTFT").
- **Batch amortization** (FlexGen) — irrelevant to a single interactive chat.

## 3. Applicability to xllama (Series S, UWP)

Constraints already on file:

- RAM budget **3801 MB** ([uwp-constraints.md](uwp-constraints.md)) — streaming
  would buy _capacity_ (models > RAM), not speed.
- **No transparent paging**: POSIX mmap is unavailable and the AppContainer
  mapping was tried and reverted with no benefit (uwp-constraints §1); GGUF is
  read into heap. Any streaming must be explicit (PR #25294 shape: expert
  cache + I/O threads over Win32 file APIs).
- **H2 MoE FAIL** ([phase7-hypotheses.md](phase7-hypotheses.md)): at
  console-admissible scale the MoE truly reads only active experts and is
  still as slow as the 3B dense — MoE brings no _speed_ headroom here even
  from RAM.

Scenarios:

| Scenario                            | Verdict                    | Why                                                                                                                                                                                                                                                 |
| ----------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense weights streamed from NVMe    | **No**                     | §2 arithmetic: sub-1 tok/s at best                                                                                                                                                                                                                  |
| MoE expert streaming (> RAM models) | **Gated on §4 + upstream** | needs measured random-read ≥ ~1.5 GB/s to clear ~1 tok/s for a plausible target (active set ~1 GB, miss rate ~2/3); also needs PR #25294 (or a port) and a console-shaped MoE catalogue entry — and H2's shadow says the win would be capacity only |
| KV/prompt-cache persistence on disk | **Yes, cheap**             | pure TTFT win, no bandwidth cliff; candidate follow-up independent of this assessment                                                                                                                                                               |

## 4. Measured: sandboxed NVMe read bandwidth (`diskbw`)

Probe: `diskbw.flag` → `diskbw-result.csv` (pattern of `membw`/`gpubw`;
`scripts/bench-diskbw.sh`). 4 GiB incompressible file — larger than the
documented 3801 MB app RAM budget, which makes full cache service _unlikely_,
not impossible (the OS file cache is not bound by the app budget) — sequential
8 MiB blocks (bulk-load shape) and random 2 MiB blocks (expert-fetch shape),
1 and 4 I/O threads. The probe prefers `FILE_FLAG_NO_BUFFERING`/`O_DIRECT`,
which bypasses the file cache outright; the `unbuffered` column records
whether that mode actually ran (it did in every row below), so the cache
question only affects the buffered fallback, which additionally drops the
cache per pass on POSIX (`posix_fadvise DONTNEED`, best-effort).

Host reference (Linux laptop, `xllama-cli --diskbw`):

| pattern | threads | unbuffered | first GB/s | best GB/s |
| ------- | ------: | ---------: | ---------: | --------: |
| seq 8M  |       1 |          1 |  1.13–2.21 |      4.47 |
| seq 8M  |       4 |          1 |  2.71–5.96 |      5.96 |
| rnd 2M  |       1 |          1 |  1.51–1.98 |      1.98 |
| rnd 2M  |       4 |          1 |  2.48–3.05 |      3.05 |

(Samsung PM9A1 512 GB, btrfs, O_DIRECT in effect; ranges span two consecutive
runs, 2026-08-18 — btrfs + background I/O make host runs noisy; the host table
is a reference, not a gate input. ⚠️ Gotcha: the first attempt ran in `/tmp`,
which is **tmpfs** on Arch — it measured DRAM (12+ GB/s "disk") and O_DIRECT
did not refuse the open. Always point `XLLAMA_DISKBW_FILE` at a real
filesystem.)

Console (Series S, Dev Mode):

| pattern | threads | unbuffered | first GB/s | best GB/s |
| ------- | ------: | ---------: | ---------: | --------: |
| seq 8M  |       1 |          1 |       2.00 |      2.03 |
| seq 8M  |       4 |          1 |       1.80 |      1.85 |
| rnd 2M  |       1 |          1 |       1.55 |      1.55 |
| rnd 2M  |       4 |          1 |       1.75 |      1.76 |

(Series S Dev Mode, package `1.5.4.900` **crossbuild**, 2026-08-18; raw CSV:
`bench/results/diskbw.csv`. Deliberate deviation from the CI-MSVC-only rule
for console benchmarks: this probe measures NVMe read bandwidth through
Win32 file APIs, not code generation — the compiler cannot move the device's
read rate, so the crossbuild number stands for the §4 gate. Any tok/s or RAM
claim would still require a CI MSVC package.) Three facts fall out:

1. **`FILE_FLAG_NO_BUFFERING` is accepted in the AppContainer** — the sandbox
   does not block unbuffered reads, so an explicit streaming engine could do
   O_DIRECT-style I/O on console.
2. **Sequential ≈ 2.0 GB/s** — ~83% of the PCIe 4.0 x2 raw spec (2.4 GB/s);
   the sandbox tax is real but small. Extra I/O threads do not help (1.8–1.85
   GB/s @4t): the device, not the submitter, is the bottleneck.
3. **Random 2 MiB ≈ 1.55–1.76 GB/s** — the expert-fetch shape loses 22–24%
   vs sequential at 1 thread and 3–5% at 4. The §4 gate (≥ 1.5 GB/s)
   **passes, marginally**.

**Read gate** (why 1.5 GB/s): a hypothetical console-admissible MoE with ~1 GB
active per token and a hot cache absorbing ⅔ of expert reads needs
`1.5 GB/s ÷ (1 GB × ⅓) ≈ 4.5 tok/s` I/O-side headroom to leave room for the
CPU side to still land ≥ 1 tok/s end-to-end. Below 1.5 GB/s random read, the
I/O alone caps the pipeline under usability before compute even starts.

### Series X follow-up (2026-11-02)

The [run sidecar](../bench/results/20261102T103941Z-series-x-streaming/run.json)
identifies Series X through Device Portal and records installed package
`1.6.1.12`; its compiler/CI provenance is unverified. The
[raw SSD probe](../bench/results/20261102T103941Z-series-x-streaming/diskbw-result.csv)
completed all four configurations with unbuffered reads:

| pattern | threads | first GB/s | best GB/s |
| ------- | ------: | ---------: | --------: |
| seq 8M  |       1 |       1.87 |      2.15 |
| seq 8M  |       4 |       1.74 |      1.74 |
| rnd 2M  |       1 |       1.44 |      1.60 |
| rnd 2M  |       4 |       1.57 |      1.58 |

This is one invocation, not a repeated-run distribution. The installed probe
hard-codes the CSV `host` field to `xbox-series-s`; preserve those raw bytes,
but use the sidecar's verified Series X identity rather than misclassifying
the run. The RAM result and its limitations live in
[uwp-constraints.md §5g](uwp-constraints.md#5g--series-x-headless-resource-probe-2026-11-02).
The temporary test file and probe flags were absent after cleanup; xllama was
restarted without changing the existing API flag, model files or settings.

## 5. Verdict and reopen conditions

**Verdict (measured): the claim is real but narrow.** The console I/O gate
passes on the number alone — 1.55 GB/s random read would give a ~4.7 tok/s
I/O-side ceiling at 1 GB active × ⅓ miss — but only marginally, and the other
two conditions (an upstream streaming engine, a console-shaped > RAM MoE worth
running) are still absent, so the recommendation below stands unchanged. Nothing
computes inside an SSD today; streamed-weights inference exists and works, at
~0.5–5 tok/s, only for sparse MoE giants with a hot cache — a capacity play.
On Series S the RAM budget makes the capacity argument tempting, but the H2
measurement, the missing upstream (PR #25294 unmerged), and the NVMe ceiling
make it a poor near-term investment. The cheap real win in this family is
KV/prompt-cache persistence (TTFT), not weight streaming.

Reopen if **any** of:

1. llama.cpp PR #25294 (or successor) merges — the explicit-streaming engine
   arrives upstream instead of needing a port;
2. a catalogue-shaped MoE with active set ≲ 1 GB and > 3.8 GB total exists
   worth running (the I/O half of this condition is already met: §4 measured
   1.55 GB/s random read on console);
3. the product accepts sub-1 tok/s for an offline "big model" mode (changes the
   gate arithmetic entirely).

## 6. Qwen3.8-27B experimental planning (Series X)

The initial research target was text-only
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
Its upstream configuration declares `Qwen3_5ForConditionalGeneration`, 64
decoder blocks and hidden size 5120. The pinned llama.cpp commit has a
`qwen35` implementation; this is architecture support, **not evidence that
this checkpoint runs on Xbox**. Vision is outside this experiment.
The dense target remains a reference experiment; §9 selects a fine-grained
MoE as the primary path toward usable large-model inference.

**Status: host-only CPU spike (§7); no console model execution.** No UWP layer-streaming engine or AirLLM Python
runtime is integrated. The existing UWP loader still allocates resident
weights (§3). Do not add this target to the shipping catalogue until the
host-smoke and console-validation ladder in [architecture.md](architecture.md)
passes. An AirLLM working-memory figure is not the checkpoint size or an Xbox
memory measurement.

`scripts/plan-layer-streaming.py` inspects the tensor descriptors of a local,
already converted text GGUF using gguf-py's read-only host mapping. It does
not download weights, execute model code, run inference or change the console.
Initialize the submodule and install its `gguf-py` dependencies in a host
virtual environment first. Pass **all** shards, each exactly once:

```bash
python3 scripts/plan-layer-streaming.py /path/to/text-model-*.gguf \
  --ram-mib "$MEASURED_APP_RAM_MIB" \
  --runtime-mib "$RUNTIME_RESERVE_MIB" \
  --read-gbs "$MEASURED_SEQ_READ_GBS" > layer-plan.json
```

The model shape check is not a checkpoint identity or integrity check: the
operator must select a trusted conversion of this target. The planner rejects
incomplete shard sets, duplicate tensors and missing decoder blocks.

Its conservative candidate policy keeps every non-block tensor resident and
reserves two buffers sized to the largest decoder block. The explicit runtime
reserve must cover KV/recurrent state, compute scratch, UI and other allocations.
`budget_fits_arithmetically` checks only this accounting; it does not demonstrate
that llama.cpp can execute with those buffers. Exit status 1 means the planned
budget does not fit; invalid inputs produce status 2.

The I/O-only decode bound divides caller-supplied sequential read bandwidth by
the sum of **all** decoder-block bytes, not the largest block. It assumes each
block is fetched once per generated token with non-block tensors resident.
Compute, repacking, synchronization and transfers can only make that candidate
pipeline slower. This is not a predicted or measured token rate. Do not reuse
the Series S measurements in §4 as Series X results.

Next gates:

1. Measure the Series X app RAM and SSD rate using `bench-ramceil.sh` and
   `bench-diskbw.sh` (these probes affect the running console app; run deliberately).
2. Inspect a trusted quantized GGUF and reject budgets that do not fit.
3. Implement bounded file reads, layer-buffer lifetime/scheduler integration and
   cancellation through a patch under `patches/`, not edits committed inside the
   llama.cpp submodule. Account for CPU repacking and persistent hybrid state.
4. Establish resident-versus-streamed logit parity on a small compatible model
   before attempting this checkpoint, then measure on-console correctness,
   memory, TTFT and decode. Planning success alone does not pass any of these gates.

Host-only planning tests (no NumPy or model weights required):

```bash
python3 -m unittest discover -s tests -p test_layer_streaming_plan.py
```

### Pinned GGUF header inspection

The initial candidate is `ggml-org/Qwen3.8-27B-GGUF`'s `Q4_K_M` text model.
The captured [descriptor report](../bench/plans/qwen38-27b-q4_k_m-header.json)
pins the repository revision and records the file size, header SHA-256, tensor
sizes and per-block byte counts. It is metadata evidence, **not** a console
benchmark or verification of the complete weights. The fetched prefix may
include the first bytes of tensor data; the full checkpoint was not downloaded.

Reproduce without third-party Python dependencies:

```bash
python3 scripts/inspect-remote-gguf.py \
  --repo ggml-org/Qwen3.8-27B-GGUF \
  --revision a316ea527fde32c12487be7574e2db5ac2ad2a7e \
  --filename Qwen3.8-27B-Q4_K_M.gguf > remote-header.json
```

The inspector accepts little-endian GGUF v3 only, limits the response body to
16 MiB (plus one byte to detect an oversized response), and rejects a server
that ignores `Range` before reading the response body. Tensor range checks use
the complete remote file size from `Content-Range`. Quantization block sizes
come from the pinned submodule's dependency-free `gguf/constants.py`; no target
model code is executed. It rejects truncated headers, unknown quantization
types, overlapping tensors and invalid data ranges.

The planner can consume the checked-in report offline, with no NumPy:

```bash
python3 scripts/plan-layer-streaming.py \
  --metadata-report bench/plans/qwen38-27b-q4_k_m-header.json \
  --ram-mib "$MEASURED_APP_RAM_MIB" \
  --runtime-mib "$RUNTIME_RESERVE_MIB" \
  --read-gbs "$MEASURED_SEQ_READ_GBS"
```

The report mode currently accepts only unsplit GGUFs; the local-reader mode
still requires every shard. Its two-buffer policy would reread all decoder
blocks per decode token, so the captured block-byte total, not an AirLLM
working-memory claim, is the relevant I/O denominator. The initial metadata-only
inspection could not reach Device Portal; the Series X follow-up in §4 has
since completed both probes.

The [console-informed candidate plan](../bench/results/20261102T103941Z-series-x-streaming/qwen38-layer-plan.json)
uses the successful headless allocation envelope as a planning cap, plus an
**unvalidated** 1024 MiB runtime reserve. Its arithmetic fit is not evidence of
scheduler support, runtime admission, correctness or model execution. At the
selected first-pass sequential probe rate, the I/O-only estimate is about
9.2 seconds per decode token; this is not measured inference latency or a
hard hardware bound. The observed SSD throughput may differ with real layer
reads, and compute/repacking/synchronization are excluded. No full model was
downloaded and no layer-streaming engine was enabled by these probes.

Additional bounded-parser and HTTP-range tests:

```bash
python3 -m unittest discover -s tests -p test_remote_gguf_metadata.py
```

## 7. Host-only CPU layer-streaming spike

There is now an opt-in C++ execution prototype in
`src/experimental/layer_streaming/`, separate from both shipping backends.
The ownership and scheduler contract is described in
[architecture.md](architecture.md#experimental-cpu-layer-streaming-runner-not-a-shipping-backend).
This is a native experiment inspired by the layer-offload approach, **not an
integration or port of the AirLLM Python package**.

### Build and validate without downloading a model

```bash
cmake -S . -B build/streaming-spike \
  -DXLLAMA_BUILD_STREAMING_SPIKE=ON \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/streaming-spike \
  --target xllama-streaming-smoke xllama-streaming-spike -j 4
ctest --test-dir build/streaming-spike -R '^streaming-smoke$' --output-on-failure
```

The normal build option is OFF. Linux CI opts in and registers the smoke as a
separate CTest test, not a new doctest case. The smoke generates tiny synthetic
four-block dense Llama and hybrid Qwen35 GGUFs, each in F32, Q4_0, Q4_K and
Q6_K (eight fixtures total). These
produce noise and are **not usable models or quality evidence**. It compares
resident and streamed logits across prefill, single-token decode and a further
two-token batch on the same persistent contexts, forcing repeated slot eviction.
It checks finite outputs at `1e-5 + 1e-5 * abs(reference)` tolerance, reduced
weight allocation, explicit budget rejection, cancellation before decode, and
file truncation followed by refusal to expose logits. Temporary fixtures are
deleted on success and retained for diagnosis on failure.

The [captured host smoke](../bench/plans/layer-streaming-host-smoke.json)
records the pinned llama.cpp revision, adapter/fixture source hashes and
per-fixture results on macOS arm64. Both the normal Debug run and a fully
ASan/UBSan-instrumented build passed all eight fixtures with zero observed
logit difference. This is not Linux CI, UWP, real-checkpoint validation or an
Xbox throughput claim. Reproduce the instrumented variant with a separate
build directory, `-DGGML_ACCELERATE=OFF`, and
`-fsanitize=address,undefined -fno-omit-frame-pointer` in both
`CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` (plus
`-fsanitize=address,undefined` in `CMAKE_EXE_LINKER_FLAGS`). Run CTest with
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

### Host runner

After obtaining a trusted, supported **unsplit** text GGUF separately:

```bash
build/streaming-spike/src/experimental/layer_streaming/xllama-streaming-spike \
  /path/to/model.gguf WEIGHT_BUDGET_MiB 'A short test prompt'
```

Replace `WEIGHT_BUDGET_MiB` with a positive integer; it caps only aligned resident
weight storage plus the two aligned layer slots, **not total process memory**.
The runner uses a raw prompt (no chat template), greedy decoding and at most
16 generated tokens with a small fixed context. This is a correctness/debug
tool, not a production Qwen chat configuration or a benchmark harness. It
reports logical tensor bytes, allocated weight bytes and actual read counters.
It does not automatically download checkpoints, upload anything, call remote
inference APIs, install packages or touch the console.

The same directory also holds `xllama-moe-streaming-spike` and its CTest
`moe-streaming-smoke`, a `qwen3moe` adaptation: dense common tensors stay
resident and each decoder layer owns a persistent expert cache of `slots`
page-aligned slabs. A whole routing step is protected before any victim is
chosen; a slot becomes valid only after every gate/up/down projection for it
has been read. Router expert counts, weights and probabilities are unchanged —
selected expert IDs are temporarily translated to cache slots for one CPU
`MUL_MAT_ID` node, then restored. Physical ubatches are one token, so prefill
and decode are compared on identical inputs; chunked prefill (one fetched expert
serving several prompt rows) is the next performance step, not part of this
correctness spike.

```bash
cmake --build build/streaming-spike \
  --target xllama-moe-streaming-smoke xllama-moe-streaming-spike -j 4
ctest --test-dir build/streaming-spike -R '^moe-streaming-smoke$' --output-on-failure
```

The [captured MoE smoke](../bench/plans/moe-streaming-host-smoke.json)
records the pinned llama.cpp revision, adapter/fixture source hashes and
per-fixture results on macOS arm64. Both Debug and ASan/UBSan builds passed
all ten synthetic `qwen3moe` scenarios (F32/Q4_0/Q4_K/Q6_K, two/four slots,
plus mixed-quant and per-node synchronization variants) with zero observed
logit difference. These fixtures produce noise, not quality evidence.

### Limits and remaining Xbox gates

- Dense `llama` and `qwen35` architectures only; no MoE, MTP, LoRA or virtual/
  transformed weight layouts. Unsupported layouts fail closed.
- CPU-only, CPU repacking disabled. No mmap, GPU offload or asynchronous
  prefetch. Every scheduler node is synchronized; two reusable slots do not
  imply overlapped read/compute.
- Buffered OS file I/O (stdio read-ahead disabled), **not** the console probe's
  unbuffered I/O path. Do not equate prototype read counters or host timing
  with the Series X SSD probe or token throughput.
- One context, one sequence and one active decode; cancellation is sticky.
  KV/recurrent state and compute scratch are additional allocations. No
  total-memory admission, file-mutation locking or SessionHub integration yet.
- **Unsplit GGUF only.** A console deployment needs shard support and the UWP
  file/provisioning constraints in [uwp-constraints.md](uwp-constraints.md).
- No UWP compilation or Xbox model execution has been performed for this
  prototype, and the complete 27B checkpoint has not been downloaded or tested.

Before shipping: implement and test a supported loader/scheduler extension via
`patches/`; add bounded shard-aware UWP reads, robust cancellation and ownership;
validate parity on a real small Qwen35 quantization (including K-quants), measure
total memory and repeated generation; only then attempt the 27B headless on
Series X. No catalogue entry or default-backend change is justified by a
synthetic smoke result.

## 8. System One / Jev: decision-shaped workloads, not faster weight streaming

The [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and [API documentation](https://docs.typesafe.ai/introduction) describe a model
trained for typed decisions and parallel question outputs rather than general
autoregressive string generation. The public
[System One adapter](https://github.com/typesafe-ai/system-one-adapter-python)
wraps external LLM APIs for comparison; it is **not Jev's inference engine or
an offline Xbox runtime**. The reviewed resources do not supply Jev weights or
a UWP implementation. Their speed/calibration claims are vendor claims on
different workloads and infrastructure, not measured xllama results.

Applicable local ideas, **not implemented by the streaming spike**:

- For one finite-choice decision, map allowed options to validated single-token
  labels and score them from the final prefill logits. This can avoid a long
  textual/JSON decode loop. Multi-token choices require sequence scoring; do
  not silently compare only their first tokens.
- Return a validated enum/boolean and let ordinary code compose decisions,
  rather than having a large model narrate every branch. Schema validity
  prevents invalid value shapes, **not semantically incorrect decisions**.
- Use a small resident model for frequent, bounded decisions; reserve the
  streamed 27B for less frequent tasks. Any routing/escalation threshold must
  be validated on held-out task data. A softmax over label logits, especially
  one renormalized over just the allowed labels, is **not a calibrated
  confidence or a reliable abstention signal**.

This changes the workload, not Qwen's architecture: ordinary Qwen cannot gain
Jev's independent multi-question parallel outputs or RLCD calibration by a
sampler flag. A finite-choice path still pays for prefill and weight reads;
the Qwen27B streaming I/O cost in §6 remains. Implementing it would be a separate
explicit decision API/mode, not a transparent acceleration of chat. No TypeSafe
API calls, SDK dependency, remote data submission, tool execution or console
changes were introduced by this assessment.

## 9. Series X large-model research direction (2026-11-02)

**Goal: usable local general-purpose model inference, not merely loading a
large checkpoint, and not replacing chat with a classifier.** The dense 27B
spike stays as a correctness/reference tool, but it is not the primary route
to this goal: its per-token read volume fails the intended interactive use.

The first research candidate is **Qwen3-30B-A3B-Instruct-2507, Q4_K_M**.
This changes the selected experiment, not the shipping catalogue. The pinned
[header capture](../bench/plans/qwen3-30b-a3b-instruct-2507-q4_k_m-header.json)
contains actual tensor descriptors, not a parameter-count estimate: 48 MoE
layers, 128 experts per layer, 8 selected. Only the bounded header prefix was
downloaded; the full checkpoint hash remains unverified.

### Primary sources reviewed and what transfers

| Source                                                                                                                                                       | Verified status / scope                                                                                                   | What to use here                                                                                                                              | What not to assume                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [llama.cpp #25294](https://github.com/ggml-org/llama.cpp/pull/25294), head `1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5`                                        | GitHub reports open, unmerged and conflicting with current upstream at review time; MIT C++ code                          | Bounded expert slots, router-ID remapping, explicit I/O, wave-partitioned prefill, one-context ownership                                      | Not a patch that can be blindly applied to our newer pin; published GPU/server numbers are not Xbox numbers |
| [ramvamp](https://github.com/y0sif/ramvamp/tree/a68b69a70027b369e20fd468d6dcefdf0e262475), source `a68b69a70027b369e20fd468d6dcefdf0e262475`                 | Rust, MIT OR Apache-2.0; source implements a 2485 MiB default expert cache, 19 slots/layer, and explicit positioned reads | Per-expert packed layout, per-layer ghost-history LFU, pin an entire routing step before eviction, reuse expert memory during chunked prefill | Linux io_uring/O_DIRECT/cgroups are not UWP APIs; its whole runtime is not an Xbox drop-in                  |
| [TurboFieldfare](https://github.com/drumih/turbo-fieldfare/tree/d994a65da9080f50d083ec22805e4f30702b5623), source `d994a65da9080f50d083ec22805e4f30702b5623` | Apache-2.0, Swift/Metal, Gemma-specific Apple runtime                                                                     | Expert-cache ownership, shared-branch/read overlap, chunked prefill, installer that never stages the full model                               | Metal kernels and Apple memory/SSD results do not carry to D3D12 or GameOS                                  |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)                                                                                               | Official model card describes hybrid attention and fine-grained MoE                                                       | Second candidate once the simpler Qwen3 MoE path is correct; potentially different recurrent/KV trade-offs                                    | A smaller active-parameter count is not itself a throughput or quality guarantee                            |

These are source/design references, not copied runtime dependencies. Any later
source port must preserve its applicable copyright/license notices. No third-
party runtime code has been imported by this research pass.

Ramvamp's pinned [README](https://github.com/y0sif/ramvamp/blob/a68b69a70027b369e20fd468d6dcefdf0e262475/README.md)
reports **2.78–2.81 tok/s at context 512**, Qwen3-30B-A3B Q4_K_M, CPU-only
Intel Core Ultra 9 185H and Micron 2400 DRAM-less QLC SSD, cold page cache,
swap disabled and a 4 GiB cgroup, using 19 slots/layer. This is the author's
measurement, **not independently reproduced here and not a Series X forecast**.
It is a credible reference for low-memory feasibility, not a transferred speed.
Older sections of that project's architecture document describe earlier cache
budgets and sessions; do not mix those with its current headline.

One tempting result was explicitly excluded as end-to-end evidence:
[discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
describes an early 4.7 tok/s result with **simulated compute**. Likewise, cache-
biased routing in some Mac demos changes the model's chosen experts. That is
not an acceptable transparent optimization of the unchanged checkpoint.

### Candidate cost and the proposed native pipeline

`inspect-remote-gguf.py` now computes a routed-expert layout for `qwen3moe` and
`qwen35moe`, requiring complete FFN projections and contiguous expert slabs on
the third tensor axis. It distinguishes the resident common core from routed
weights and separately reports a proposed 4 KiB-aligned packed expert stride.
It does not repack or run the model.

For the pinned candidate, keeping common weights resident reduces uncached
expert reads to about **1.10 GB per decode token**, rather than reading every
decoder block as the dense spike does. This still needs cache hits and efficient
compute. A per-layer expert file under the proposed layout fits the per-file
constraint; all exact sizes and source offsets are in the captured report.
The layout packs gate/up/down slabs for one expert together without changing
their original quantized bytes or the router's decisions.

The executable [cache scenario](../bench/plans/qwen3-30b-a3b-cache-scenario.json)
uses a **proposed**, conservative 4096 MiB total-memory planning cap, 16 slots
per layer, an **unmeasured** 768 MiB runtime reserve, and a **hypothetical**
50% byte-weighted cache-hit ratio. The SSD-rate input comes from the Series X
random-read probe, but its transfer to this expert geometry is unmeasured.
The arithmetic fits; it does **not** establish actual memory admission,
cache effectiveness, TTFT or a token rate. Do not turn the I/O-service estimate
into an end-to-end throughput prediction.

```bash
python3 scripts/plan-moe-cache.py \
  bench/plans/qwen3-30b-a3b-instruct-2507-q4_k_m-header.json \
  --ram-mib 4096 --runtime-mib 768 --slots 16 \
  --read-gbs 1.57 --assumed-byte-hit-rate 0.5
python3 -m unittest tests/test_moe_cache_plan.py
```

Implementation order:

1. **CPU correctness first:** resident attention/router/embedding/norms/output;
   bounded expert slots per layer; read only exact router-selected misses.
   Preserve the complete requested expert set before choosing eviction victims;
   never reuse a buffer until both its read and queued compute have completed.
2. **Portable storage then UWP:** pack expert slabs contiguously, bounded and
   hash-verified during installation; translate positioned/direct I/O to
   AppContainer-compatible Win32 reads. The prototype's `MoeFile` implements
   this for Win32/UWP (`CreateFile2`/`ReadFile`, optional `FILE_FLAG_NO_BUFFERING`)
   and POSIX (`pread`, optional `O_DIRECT`), falling back to buffered reads and
   reporting the achieved mode rather than pretending a direct read happened.
   Test alignment, short reads, cancellation and cache-mode reporting rather
   than silently treating buffered reads as direct reads.
3. **Prefill is a first-class gate:** process tokens in chunks/waves so one
   expert read serves several prompt rows. Single-token streamed prefill may
   generate correct text and still fail usability. Sweep chunk size and thread
   count on-console, with all staging counted in the memory cap.
4. **Cache/overlap experiments:** compare LRU and ghost-history LFU on recorded
   Xbox routing traces; overlap only guaranteed hit/shared compute with misses.
   Do not assume a borrowed hit rate or predictable future-layer experts.
5. **GPU only where measured:** extend/use the existing D3D12 quantized matmul
   work after profiling. The current backend does not implement expert
   `MUL_MAT_ID`; enabling GPU layers is not enough. CPU and GPU caches also
   consume the same constrained physical-memory system, not two budgets that
   can simply be added together.

The synchronous per-node callback used in the dense host spike is a parity
instrument, **not the final performance architecture for this MoE pipeline**.
Prefer supported, narrowly scoped loader/scheduler hooks under `patches/` to
rewriting every model's forward pass or repurposing the debug adapter in UWP.

### Proposed acceptance gates, not promises

- **Local-only:** no remote inference; same quantized model bytes and routing
  as the resident reference. Chat remains general-purpose, not just enum output.
- **Correctness:** synthetic MoE parity, then real small-MoE logits/task checks
  against resident llama.cpp before the 30B experiment; no silent router bias,
  dropped experts, changed sampling or low-bit quality shortcut.
- **Memory:** measure the complete process peak (common weights, cache,
  KV/recurrent state, scratch, UI and copies). An arithmetic fit or the heap
  probe alone cannot pass this gate.
- **Usability target:** at least **2 tok/s** sustained decode and first visible
  token within **30 seconds** for an exactly tokenized 128-token prompt at
  context 512, including model load but excluding a first-time model download.
  Score at least three fresh-process runs generating up to 128 tokens; report
  every run, median, stop reason, full latency and peak memory. Repeat at context
  2048 to expose long-chat degradation; do not claim the context-512 rate holds.
- **Honest failure:** if the target fails, profile and report it, reduce the
  model/adjust the mode or label it an offline experiment. Do not promote the
  checkpoint to the catalogue by relaxing the benchmark after seeing the result.

Status: **descriptor/cache planning and a host-native MoE prototype (§10)
implemented; real-model correctness, UWP integration and Xbox 30B inference
remain pending**.
AirLLM-style dense streaming, Jev's decision-only workload, server-sized batching,
and blindly aggressive quantization are not the primary path to this goal.

## 10. Host-native MoE cache and execution prototype

`src/experimental/layer_streaming/` now contains a native C++ `qwen3moe`
execution prototype, not just the analytical planner. It is behind the existing
default-OFF `XLLAMA_BUILD_STREAMING_SPIKE` option, and is **not linked into UWP,
the GUI, SessionHub or the LAN server**. The implementation is original code
using the reviewed cache strategy; no third-party runtime was vendored.

### Build and run the correctness checks

```bash
cmake -S . -B build/streaming-spike \
  -DXLLAMA_BUILD_STREAMING_SPIKE=ON \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/streaming-spike \
  --target xllama-moe-streaming-smoke xllama-moe-streaming-spike -j 4
ctest --test-dir build/streaming-spike -R '^moe-streaming-smoke$' --output-on-failure
```

`moe-streaming-smoke` is a separate CTest, not a doctest case. It exercises
the cache independently, then constructs ten synthetic `qwen3moe` scenarios:
F32, Q4_0, Q4_K and Q6_K at two/four slots per layer; a conservative per-node
Q4_K case; and a mixed K-quant case. The mixed fixture includes Q8_0/Q5_K
common projections, a Q6_K output head, and different Q4_K/Q6_K expert-down
slabs across layers. These fixtures generate noise, not quality evidence.

Every model scenario compares all vocabulary logits with an independently
loaded resident llama.cpp model through prompt prefill, incremental decode
and another multi-token input on persistent contexts. Physical ubatches match
at one token, so the comparison does not conflate changed kernels/batching
with streaming. It checks finite values at
`1e-5 + 1e-5 * abs(reference)` tolerance and forces real cache hits, misses and
evictions. Expert read counters must start at zero after model/context creation;
the reduced allocation check covers **weight storage**, not whole-process RAM.

Additional checks cover whole-step protection (including hits listed after a
miss), pending-slot ownership, ghost frequency surviving eviction, duplicate
IDs, invalid requests without telemetry changes, top-k/weight-budget rejection,
cancellation before expert reads, and a truncated model causing a short-read
failure. Failed contexts refuse both logits and further decode. The cache is
single-owner; no slot is released until all projections have completed their
reads and all three matmuls have synchronized.

The [captured host result](../bench/plans/moe-streaming-host-smoke.json) records
adapter/cache/fixture source hashes and the pinned llama.cpp revision. All ten
synthetic scenarios passed on macOS arm64 in Debug and in a fully
ASan/UBSan-instrumented build, with zero observed logit difference. The paired
Q4_K checkpoint/per-node scenarios also preserve identical cache access and
read counts while requiring fewer synchronization checkpoints. Those are
correctness and work-count observations, **not an end-to-end speedup**, a
real-model quality result or evidence of Xbox performance.

### Host runner and current limits

```bash
build/streaming-spike/src/experimental/layer_streaming/xllama-moe-streaming-spike \
  /path/to/unsplit-qwen3moe.gguf WEIGHT_BUDGET_MiB SLOTS_PER_LAYER 'Raw prompt'
```

Both numeric arguments are explicit positive integers. The weight budget
admits aligned common weights plus all per-layer cache allocations **before
reading them**, but excludes metadata, KV, compute scratch and runtime memory.
It is not the total-memory cap proposed in §9. The runner uses a raw prompt,
greedy decoding and at most 16 output tokens; it is not the production chat
template, sampler or a performance benchmark. It reports common/cache/logical
weight sizes, expert bytes actually read, routing decisions, hits, misses and
evictions. The physical ubatch is one even when the logical input contains
several tokens.

The ownership/ID-restoration/checkpoint contract lives in
[architecture.md](architecture.md#experimental-cpu-layer-streaming-runner-not-a-shipping-backend).
Pure-float nodes are grouped only between integer/expert checkpoints; the
conservative all-node synchronization mode remains a correctness comparator,
not a recommended speed mode.

Remaining limitations:

- Host CPU only; pinned private metadata-only model promotion and tensor-axis
  rebinding. No public loader ownership contract or production scheduler hooks.
- Unsplit GGUF, separate gate/up/down projections, unchanged router and
  probabilities. Unexpected virtual/transformed/expert views are rejected.
- Synchronous source-GGUF slab reads; no packed install format, async I/O,
  direct/unbuffered Win32 reads or Windows/UWP validation. Disabling stdio
  read-ahead does **not** bypass the OS file cache.
- Single owner/context/sequence. Sticky cancellation is checked at scheduler
  boundaries and read chunks; it does not interrupt an already blocking read.
- Token-at-a-time physical prefill: no wave/chunked expert reuse yet. This can
  preserve correctness and still fail the TTFT gate.
- No file-mutation lock, total-memory admission, D3D12 expert kernel, real-model
  quality validation, full 30B download or Xbox 30B generation has occurred.

This closes the native-cache/host-execution bring-up step only. It does not
close the usability gates in §9, justify a catalogue entry, or establish a
Series X token rate. Next: supported loader/scheduler hooks and bounded,
shard/packed-aware UWP I/O, then real-model parity and measured prefill/decode.

## 11. UWP smoke probe: `moestream.flag` on Xbox

A headless smoke probe is now wired into the UWP app so you can validate that a
GGUF model loads and generates on console **before** attempting the MoE
streamer integration. It uses the existing resident `run_inference` path
(llama.cpp backend), so no submodule edits are required.

### Build (Windows host, MSVC + UWP workload)

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64 -Backend llamacpp
```

The `llamacpp` backend is required; `ort`/`unified` will run the model through
ORT GenAI instead. `-Backend llamacpp` is the shipping bench lane and the one
used for GGUF smoke tests.

### Deploy to the console

```bash
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
```

### Copy a model into LocalState

The probe reads `LocalState\models\<model_name>`. Copy a trusted GGUF there
(Device Portal file upload, or `deploy.sh upload-file`):

```bash
./scripts/deploy.sh upload-file /path/to/model.gguf <PFN> "" "models\model.gguf"
```

### Run the smoke probe

Create `moestream.flag` whose first line is the model filename, then launch the
app. The app reads the flag, deletes it, runs one prefill+decode pass, and
writes `LocalState\moestream-result.csv` plus `moestream-result.csv.done`.

```bash
#From a PowerShell or bash session with XBOX_IP / XBOX_USER / XBOX_PASS set:
echo "lfm25-350m" > meestream.flag   # or your GGUF filename
./scripts/deploy.sh upload-file meestream.flag <PFN> "" "moestream.flag"
./scripts/deploy.sh start-app
```

Wait for `moestream-result.csv.done` (poll with `deploy.sh fetch-file`), then
fetch the CSV:

```bash
./scripts/deploy.sh fetch-file <PFN> moestream-result.csv ./moestream-result.csv
cat ./moestream-result.csv
```

The CSV contains `success`, `t_p_eval_ms`, `t_gen_ms`, `tokens_generated` and
any `error_msg`. `success=1` means the model loaded and generated on console;
`success=0` with an error message tells you why it failed (missing file, wrong
backend, out of memory, etc.).

### What this validates and what it does not

- **Validates:** GGUF file is readable in the AppContainer sandbox, the llama.cpp
  backend loads the model on Xbox, prefill+decode complete, and the result CSV
  is written to LocalState.
- **Does not validate:** MoE expert streaming, cache hit rates, the 30B model
  at interactive speed, or any UWP integration of `MoeStreamingModel`. Those
  require a separate Windows/MSVC build with the host `layer_streaming`
  adapter ported through `patches/` against the pinned llama.cpp.

Status: **UWP smoke probe implemented; real-model MoE correctness, UWP MoE
integration and Xbox 30B inference remain pending**.
