# Native Bielik 3.2

[Back to README](../README.md) · [API reference](api.md) · [Serving](serving.md)

The explicit `--native-tokens` profile serves original Bielik 3.2 weights using
existing answer tokens. It loads no LoRA, extra decision heads, vocabulary
extension or heads plugin. For installation, see the [quick start](../README.md#quick-start). Use vLLM
0.32.0 in the same environment as the launcher. Both commands run in the
foreground: start the backend and frontend in **separate terminals**.

Chat tool-call and reasoning parsers are **required**, including for middleware.
First configure the [external bielik-tools checkout](serving.md#tools-and-reasoning-for-bielik-adapter-chat)
through `BIELIK_TOOLS_DIR` or backend `--bielik-tools-dir`. Missing parser files
fail startup; no parser code is copied into this project.

```bash
bielik-one backend /path/to/bielik-3.2 --native-tokens
bielik-one frontend /path/to/bielik-3.2 --native-tokens
```

The backend serves `base` on loopback port 8025; the frontend exposes typed
`/v1/systemone` requests as `decision` on port 8026 and proxies ordinary chat as
`base`. One copy of GPU weights serves both routes. Native checkpoints need not
contain decision-package checksum manifests; `--dry-run` reports
`checksums_verified: false`. Pin the checkpoint revision for reproducibility.
Keep the raw backend on loopback unless your network is trusted.

Choice uses the existing training-label order: `A..Z`, `a..z`, followed by the
uppercase pairs supported as single tokens by the tokenizer. The code uses at
most 255 symbols. Reads exceeding 128 requested token IDs reuse the identical
complete prompt in multiple requests; raw vocabulary logprobs are merged and
normalized together. Usage counts every physical chunk prompt. Noul reads `no`
and `yes`; score reads the digits of ordered levels. Labels always expand to
independent noul questions, retaining the labels response format and a 0.5
threshold. Label probabilities need not sum to one and the selected set may be
empty. These probabilities are uncalibrated unless the model provides calibration.json.

Native decisions use the supplied instructions in ChatML without a general-chat
preamble. The default assistant suffix is `<think></think>`; frontend option
`--decision-prefix plain` omits it. This reads answer probabilities, not generated
reasoning or JSON. Ordinary chat retains the model's original template.

Request fields `read_twice` and `read_twice_max_input_tokens` control repetition.
Set `read_twice: true` with `read_twice_max_input_tokens: null` to repeat every
input that fits the context. The threshold is decided once per question across
candidate/permutation reads. Context overflow returns 422 without truncation.
Answers report the effective prompt variant, reason and read counts;
`prompt_details: true` includes read lengths and backend chunk counts.

The shared read pool and queue are bounded across HTTP requests, including
labels expansions. Defaults are 8 read workers, 64 queued reads, 32 active
processing requests, a 180-second backend timeout and a 300-second processing
deadline including admission. Configure them on the native frontend with `--max-num-seqs`,
`--read-queue-size`, `--max-requests`, `--backend-timeout` and `--request-timeout`.
Admission overload returns 503 with `Retry-After: 1`; timeouts return 504.
Disconnect cancellation stops queued/future reads; already running synchronous
calls retain their slots until completion or timeout. Existing body limits and
error formats apply. `--read-queue-size`, `--max-requests` and `--backend-timeout` are
rejected for other profiles; `--request-timeout` applies to every frontend.

BF16 is the default. `--fp8` enables on-load FP8 with an unquantized lm_head.
Prequantized FP8 checkpoints are detected by vLLM without this flag. Prefix
caching is enabled; the default context is 32768. `--kv-cache-gb` and `--eager`
remain available. FP8 can change probabilities and decisions, including under
batching/cache changes. Transformers and vLLM BF16 can also differ numerically;
backend agreement is not a benchmark accuracy result.

## Single-server middleware

The native middleware adds `/v1/systemone` directly to vLLM, sharing one port and
one copy of the original weights with chat. No separate frontend, LoRA or heads
plugin is required. Chat tool-call and reasoning parsers are mandatory; the
complete command below includes them:

```bash
export BIELIK_TOOLS_DIR=/path/to/bielik-tools
export SYSTEMONE_MODE=native
export SYSTEMONE_PACKAGE=/path/to/bielik-3.2
export VLLM_PLUGINS=""
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve "$SYSTEMONE_PACKAGE" --served-model-name bielik-base \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --max-logprobs 128 --logprobs-mode raw_logprobs \
  --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware \
  --enable-auto-tool-choice \
  --tool-parser-plugin "$BIELIK_TOOLS_DIR/tools/bielik_vllm_tool_parser.py" \
  --tool-call-parser bielik \
  --reasoning-parser-plugin "$BIELIK_TOOLS_DIR/tools/bielik_vllm_reasoning_parser.py" \
  --reasoning-parser bielik \
  --exclude-tools-when-tool-choice-none
```

Use `model: "bielik-base"` for decisions and ordinary chat. The model/tokenizer
paths must match `SYSTEMONE_PACKAGE`. Startup validates vLLM's resolved model,
context length, dtype and logprob mode, refusing LoRA and unsupported
quantization. Supported modes include BF16, on-load FP8 and prequantized
`compressed-tensors` with FP8 weights and optional FP8 activations. Set
`SYSTEMONE_DECISION_PREFIX=plain` to omit the default empty-think suffix.

`GET /v1/systemone/health` checks backend/model readiness. vLLM authentication,
CORS, shared internal-token/TLS settings and restricted management/prompt-logprob
routes work as for the [adapter middleware](serving.md#single-server-middleware-adapter-packages). When exposing the service,
set `--api-key` and restrict non-`/v1` routes (including `/invocations`) with a
reverse proxy or network policy: vLLM API-key authentication does not protect
every route. Client connections to an exposed service must use HTTPS: configure
both `--ssl-keyfile` and `--ssl-certfile`, or terminate TLS at that reverse
proxy. The default middleware mode remains `adapter`.

Native processing bounds use `SYSTEMONE_WORKERS` (8),
`SYSTEMONE_READ_QUEUE_SIZE` (64), `SYSTEMONE_MAX_REQUESTS` (32),
`SYSTEMONE_BACKEND_TIMEOUT` (180 seconds) and `SYSTEMONE_REQUEST_TIMEOUT`
(300 seconds; it also applies in adapter mode). These reuse the frontend's bounded pool, overload errors,
deadlines and cooperative cancellation. The actual vLLM context limit is
reflected in prompt metadata and context errors.

## Calibration

Store model-specific `calibration.json` beside the original checkpoint's
`config.json`. The native frontend and native middleware load it automatically.
The library contains no hard-coded model temperatures. Without the file,
probabilities remain uncalibrated. Invalid or explicitly incomplete calibration
files fail startup instead of silently falling back.

A single table applies to **all inference variants**: standard and read twice,
read-twice thresholds, both decision prefixes, questions before state, option
permutations, and BF16 or opt-in FP8. Fitting configuration records provenance,
not inference restrictions. This policy does not establish equal calibration
quality for every variant. Explicit request `temperature` overrides saved values.
Labels expanded to noul use their own `labels` temperature; ordinary noul uses
`binary`. Existing adapter and merged-package calibration behavior is unchanged.

For an explicit frontend file:

```bash
bielik-one frontend /path/to/bielik-3.2 --native-tokens \
  --native-calibration /path/to/calibration.json
```

Use `--no-calibration` to disable loading. With native middleware, set
`SYSTEMONE_NATIVE_CALIBRATION=/path/to/calibration.json` for an explicit file,
or `SYSTEMONE_NO_CALIBRATION=1` to disable loading. Explicit files combined with
disabled calibration are rejected. Health reports `calibrated` and `calibration`;
these describe loaded defaults even if a request overrides the temperature.

The JSON `temperatures` object must contain four positive finite numbers:
`choice`, `binary`, `score`, and `labels`. Optional `configuration` records fitting
provenance. `complete_validation_collection: false` marks a provisional file and
prevents loading it. Calibration values belong with model weights, not in the
library. Temperature scaling runs in the API on CPU without changing weights or
requiring another model pass.

## Chat tools and reasoning

The launcher and the complete middleware command above both load the required
upstream `bielik-tools` parsers. The original Bielik 3.2 chat template and decision
prompts stay unchanged. Use `BIELIK_TOOLS_DIR` to select the external checkout;
backend `--bielik-tools-dir` overrides it. Both parser files must be compatible
with vLLM 0.32.0. There is no parser opt-out.

These settings apply to ordinary chat: tool calls are returned in `tool_calls`
and thinking output is separated into the reasoning field. For Bielik 3.2,
`reasoning_effort: "none"` disables thinking; a supported non-none effort enables
it. Keep token budgets large enough to complete thinking and the final answer.
The typed decision endpoint continues reading the configured answer tokens.
