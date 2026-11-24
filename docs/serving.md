# Serving

[Back to README](../README.md) · [API reference](api.md) · [Native Bielik 3.2](native.md)

This guide covers the published adapter and merged decision packages. For original
Bielik 3.2 checkpoints, use the [native token guide](native.md); their token-read
settings differ from those of the decision packages.

## Adapter and merged packages

Python 3.12+ and vLLM 0.32.0 are required for NVIDIA serving. Install the runtime:

```bash
pip install "bielik-one[vllm] @ git+https://github.com/speakleash/bielik-one"
```

For `*-Adapter` packages, also install the heads plugin in the same environment:

```bash
pip install --no-deps "systemone-vllm-heads @ git+https://github.com/speakleash/bielik-one#subdirectory=vllm-heads-plugin"
```

Start the backend and, in a second terminal, the API, with the model's Hugging Face repo id:

```bash
bielik-one backend speakleash/Bielik-One-11B-v3.0-r1
bielik-one frontend speakleash/Bielik-One-11B-v3.0-r1
```

The package is downloaded to the Hugging Face cache on first use (`--revision` pins a commit). A local directory, for example from `hf download ... --local-dir model`, works the same: `bielik-one backend model`. `bielik-one backend <package> --dry-run` verifies the published runtime checksums and prints the exact vLLM command. Startup verification skips the root-level `README.md`, `LICENSE`, `.gitattributes` and `requirements.txt`, so editing or omitting these auxiliary files does not block inference. All other manifest entries, including every weight file, remain fully verified for both the backend and frontend. The launcher detects the package type and uses the profile validated for it:

| Package | Backend | API |
|---|---|---|
| `Bielik-One-11B-v3.0-r1-Adapter`, `Bielik-One-11B-v3.1-r1-Adapter`, `Bielik-One-7B-v3.0-r1-Adapter` | vLLM + LoRA + heads plugin | `http://127.0.0.1:8026` |
| `Bielik-One-11B-v3.0-r1`, `Bielik-One-11B-v3.1-r1`, `Bielik-One-7B-v3.0-r1` | stock vLLM | `http://127.0.0.1:8026` |
| `Bielik-One-4B-r1-Adapter` | vLLM + LoRA + heads plugin | `http://127.0.0.1:8012` |
| `Bielik-One-4B-r1` | stock vLLM | `http://127.0.0.1:8012` |

**Adapter packages are a small add-on.** They share the unchanged base weights, so if you already serve the base model (for example `speakleash/Bielik-11B-v3.0-Instruct`) with vLLM, adding a decision adapter costs only its LoRA weights and heads: about 1.0 GB for the 11B models, 1.4 GB for the 7B model and 0.7 GB for the 4B model. The same engine keeps serving ordinary chat with the original base model. Merged packages need their own full copy of the weights.

## GPU memory and performance

The backend runs in **bfloat16** by default, the precision of all reported results and of the calibration. An 11B model then needs about 30 GB of GPU memory (weights plus the configured KV cache), a 7B model about 22 GB and the 4B model about 12 GB. On smaller GPUs add `--fp8` (on-load FP8 weight quantization, BF16 language-model head): it roughly halves weight memory and raises throughput (1.4x measured for the 4B model), but changes 2-18% of decisions near the decision boundary depending on model and question type, and the calibration was not fitted for it.

The backend uses CUDA graphs, which speed up chat generation and leave decisions unchanged; `--eager` turns them off for a shorter start or if graphs cause problems on a GPU.

For many concurrent users, `--max-num-seqs N` (pass it to both the backend and the frontend) and `--kv-cache-gb GB` raise the validated profile, for example `--max-num-seqs 32 --kv-cache-gb 4` for `Bielik-One-4B-r1` (about 14 GB of GPU memory instead of 10.5 GB).

For throughput, send all questions about one state in one request: the state is read once and the other questions reuse the prefix cache. Use `--port` / `--backend-url` to move ports.

## Tools and reasoning for Bielik adapter chat

Tool-call and reasoning parsers are **required** for vLLM adapter serving.
For Bielik 3.0, 3.1 and Minitron adapters (and original Bielik 3.2), obtain one
trusted upstream [bielik-tools](https://github.com/speakleash/bielik-tools) checkout
compatible with vLLM 0.32.0:

```bash
git clone https://github.com/speakleash/bielik-tools.git
export BIELIK_TOOLS_DIR="$PWD/bielik-tools"
```

Pin that checkout to a revision you have tested. The variable points to the
repository root containing `tools/`, not to `tools/` itself. Keep this variable
available in the backend's environment:

```bash
bielik-one backend speakleash/Bielik-One-11B-v3.1-r1-Adapter --dry-run
```

Alternatively pass `--bielik-tools-dir /path/to/bielik-tools`; it overrides the
environment. The launcher always includes `--enable-auto-tool-choice`, both
`bielik` parsers via `--tool-parser-plugin` / `--reasoning-parser-plugin`, and
`--exclude-tools-when-tool-choice-none`. Missing checkout files fail before
starting vLLM. `--no-bielik-chat-parsers` is rejected. No parser files are copied,
downloaded or maintained in this repository; parser code runs from that external
checkout inside vLLM.

Supported adapter bases are `speakleash/Bielik-11B-v3.1-Instruct`,
`speakleash/Bielik-11B-v3.0-Instruct` and
`speakleash/Bielik-Minitron-7B-v3.0-Instruct`. Bielik 3.1 keeps its original chat
template; use `chat_template_kwargs={"reasoning_effort": "medium"}` (or `none`,
`low`, `high`, `max`). The 3.0 bases use upstream
`tools/bielik_advanced_chat_template.jinja` for chat; use
`chat_template_kwargs={"enable_thinking": true}` and `skip_special_tokens: false`.
Decision prompts still use the package's pinned template.

Qwen adapters automatically use the built-in `qwen3_coder` tool parser and
`qwen3` reasoning parser, following the [vLLM Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md).
They do not use the Bielik checkout. The environment variable is ignored for
Qwen, merged packages, the frontend and MLX; an explicit `--bielik-tools-dir` on
an unsupported model or non-vLLM-backend command is rejected.

Both `--shell` and `--dry-run` include all required parser flags. Keep them when
running vLLM yourself or using middleware. Middleware refuses startup without the
required tool/reasoning parser configuration.

## Running vLLM yourself

`bielik-one backend` only verifies the package and starts an ordinary `vllm serve` with a validated set of flags; the decisions are computed by `bielik-one frontend`. To run vLLM under your own process manager or container, print the exact command and start it yourself:

```bash
PACKAGE=/path/to/model
BACKEND_URL=http://127.0.0.1:8025  # Use the host/port from the printed command.
bielik-one backend "$PACKAGE" --shell    # One executable shell command.
bielik-one backend "$PACKAGE" --dry-run  # JSON with "command" and "env".
bielik-one frontend "$PACKAGE" --backend-url "$BACKEND_URL"
```

Both verify the checksums and start nothing. `--port`, `--backend-host`, `--max-num-seqs`, `--kv-cache-gb`, `--eager` and `--fp8` are reflected in the printed command.

Requirements: vLLM 0.32.0, and for `*-Adapter` packages the `systemone-vllm-heads` plugin installed in the same environment.

Keep these as printed, because the decision reads depend on them:
- the environment: `VLLM_PLUGINS` and `SYSTEMONE_HEADS_MANIFEST` (adapter packages), `VLLM_MAX_NUM_BAD_WORDS` (merged packages) and `VLLM_USE_V2_MODEL_RUNNER=0`;
- the model, its `--revision` and `--hf-overrides` architecture; the LoRA module name `decision` and its rank and target modules;
- `--served-model-name` (the frontend addresses these aliases);
- `--max-logprobs 256` and `--logprobs-mode processed_logprobs`;
- `--dtype bfloat16` and `--max-model-len 32768` (the calibration and the reported results are bfloat16, with no truncation).

Free to change for your hardware and load: `--host`, `--port`, `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--max-num-seqs` (pass the same value to the frontend), `--max-num-batched-tokens`, `--enforce-eager`, and general vLLM options such as logging. Do not set `--api-key`: the frontend does not send one. Keep the backend on a private network instead; it accepts internal decision flags.

The frontend checks every read: a backend started without the required logprob settings is refused with an error instead of returning wrong probabilities. After starting both, `curl <frontend>/health` should report `"status": "ok"` and `"calibrated": true`.

### Single-server middleware (adapter packages)

The middleware contributed by [Filip Reka](https://github.com/FIlipHand) adds `/v1/systemone` to the same vLLM server as chat, without a separate frontend process. It supports `*-Adapter` packages and uses the existing decision engine, calibration and package checks, including weight checksums.

Configure the [required chat parsers](#tools-and-reasoning-for-bielik-adapter-chat),
then use a downloaded adapter package and print the validated backend command:

```bash
export SYSTEMONE_PACKAGE=/path/to/adapter-package
bielik-one backend "$SYSTEMONE_PACKAGE" --shell
```

Run the printed command with `--middleware bielik_one.api.vllm_middleware.SystemOneMiddleware` appended. Keep its printed environment and model/LoRA flags, including `SYSTEMONE_HEADS_MANIFEST`. Do not start `bielik-one frontend` for this mode. vLLM 0.32 has its own `POST /v1/systemone` (generic structured decisions, disabled for the SystemOne architectures); the middleware replaces that route. Both chat and decisions use the base model's public served name (for example `bielik-base`); the internal `decision` LoRA name is hidden from `/v1/models` and rejected in public requests; SageMaker adapter headers and vLLM's runtime LoRA management routes are refused for clients, so the adapter cannot be selected, unloaded or loaded again under another name. If you register the adapter under a different `--lora-modules` name, set `SYSTEMONE_LORA_NAME` to it.

Decisions use vLLM's own authentication and CORS middleware, including API keys configured through `--api-key`, its config-file equivalent or `VLLM_API_KEY`. `SYSTEMONE_MAX_BODY_MB` limits decision and public request bodies except form uploads (default 16 MiB). With `--api-server-count` above 1 (vLLM's default with `--data-parallel-size` above 1), set the same secret `SYSTEMONE_INTERNAL_TOKEN` in each process; the server refuses to start without it. For TLS, Unix sockets or a different loopback address, set `SYSTEMONE_SELF_URL` explicitly (with TLS, use a host name in the certificate, and `SYSTEMONE_SELF_CA=/path/ca.pem` when the certificate is not signed by a system-trusted CA). See the middleware module's docstring for all settings.

Use `GET /v1/systemone/health` as the readiness check of decisions: it answers 200 with the served model names and calibration once decisions work, and 503 (or 500 for a missing setting) with the reason otherwise; it is under `/v1`, so it needs the API key when one is set. vLLM's own `/health` only reports the engine. After deploying, `python scripts/check_middleware.py --url <server> --api-key <key> --model <served name>` checks decisions, the readiness route, authentication and the LoRA and backend-flag filters, and exits non-zero on any failure. vLLM API-key authentication does not protect every route: restrict routes such as `/invocations`, `/health` and `/metrics` in a reverse proxy before exposing the port. Client connections must use HTTPS, with both `--ssl-keyfile` and `--ssl-certfile` or TLS termination at the proxy.

## Apple silicon (MLX)

On a Mac with Apple silicon, adapter packages run on [MLX](https://github.com/ml-explore/mlx) instead of vLLM. As with the vLLM adapter backend, one copy of the base model serves both ordinary chat (adapter off) and the decisions (LoRA and decision heads on), chosen per request:

```bash
pip install "bielik-one[mlx] @ git+https://github.com/speakleash/bielik-one"
bielik-one backend speakleash/Bielik-One-4B-r1-Adapter --engine mlx
bielik-one frontend speakleash/Bielik-One-4B-r1-Adapter
```

The frontend is the same as for vLLM, and so are the API, the ports and the model names. The backend (`bielik_one/mlx_backend.py`) answers the requests the frontend sends to vLLM: one-token reads under `decision` with the LoRA on and the decision heads (float32) as logits, and generation by the unchanged base model under the base alias (`qwen-base`, `bielik-base`) on `/v1/chat/completions` and `/v1/completions`, streaming or not. The LoRA is not merged into the base weights; it is switched on only while a decision read runs. One forward pass runs at a time; a chat generation releases the GPU between tokens, so decisions are not blocked by a long answer. Generation takes `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens` and `stop`; the vLLM parameters it does not implement (`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `min_tokens`, `ignore_eos`, `logit_bias`, `bad_words`, `stop_token_ids`, `allowed_token_ids`, `n` > 1) are refused with a 400 unless they have their neutral value, rather than silently ignored.

The base model is the repository and revision the package pins (`heads.json`, `serving.json`). It is downloaded to the Hugging Face cache, or taken from a local directory with `--base DIR` (for example from `hf download <base repo> --revision <pinned revision> --local-dir DIR`). Its weights, config and tokenizer files are checked against the Hub's checksums of the pinned revision before loading, and such files the pinned revision does not have (an extra `model*.safetensors` or `chat_template.jinja`, for example) are refused; `--skip-base-check` skips this (offline use), with a warning. `--cache-limit-gb` caps the MLX buffer cache (default 4). `--dry-run`, `--port` and `--backend-host` work as for vLLM; the vLLM options (`--fp8`, `--eager`, `--max-num-seqs`, `--kv-cache-gb`) are refused. Merged packages are not supported by the MLX backend; their weights would need a conversion first.

Pinned versions: `mlx==0.32.4`, `mlx-lm==0.32.2`, bfloat16 weights. On 600 public validation-style questions (Tacit dev, 4 languages, `choice`/`noul`/`score`, raw probabilities), Bielik-One-4B-r1 on this backend agreed with the merged package converted to MLX on 99.3% of decisions (accuracy 91.7% vs 91.3%), using 8.2 GiB of weights for chat and decisions together instead of two models. With the same prompts on both sides, decisions agreed with the vLLM adapter backend on 99.5% (4B), 99.3% (11B-v3.0) and 99.5% (11B-v3.1). Greedy chat is token-identical to plain `mlx_lm` generation with the base model.

For MLX-specific LoRA and head math tests, work from a source checkout on
Apple silicon. A Git URL installation alone does not leave the test suite in
your current directory:

```bash
git clone https://github.com/speakleash/bielik-one.git
cd bielik-one
pip install -e ".[test,mlx]"
pytest -q
```

These tests are skipped when MLX is not installed.

## Deployment and untrusted text

**Pinning the package.** Without `--revision`, the launcher downloads the repository's current `main`, and `SHA256SUMS.json` only proves that the files match the manifest that came with them; a package republished with a matching manifest passes. For production, record two values once from a source you trust (the `--dry-run` output of a reviewed download prints both as `revision` and `manifest_sha256`) and start with them: `bielik-one backend speakleash/Bielik-One-4B-r1 --revision <40-character commit id> --manifest-sha256 <hex> --require-pinned-revision`, the same for `frontend`. `--manifest-sha256` refuses a package whose manifest differs, which pins every file the manifest lists, and `--require-pinned-revision` refuses to start without a full commit id (or, for a local directory, without `--manifest-sha256`). The vLLM middleware takes the manifest digest as `SYSTEMONE_MANIFEST_SHA256`. Original native checkpoints (`--native-tokens`, `SYSTEMONE_MODE=native`) have no manifest, so only `--revision` pins them.

Both processes bind to `127.0.0.1` by default. `--host` moves only the frontend (typed API and chat proxy); the raw vLLM backend stays on loopback unless you pass `--backend-host`, because it accepts internal decision flags. The separate frontend has no authentication: to expose it, put it behind a reverse proxy that authenticates clients and terminates TLS for HTTPS. Requests are limited to 64 questions, 16 permutations, 400,000 characters of state and 400,000 characters of questions. The frontend refuses a body larger than 16 MiB with HTTP 413 before reading it whole (`--max-body-mb` on `bielik-one frontend` changes it); text that is not valid Unicode (a lone UTF-16 surrogate such as `"\udead"`) is refused with 422. Every response carries an `x-request-id` header: the client's `X-Request-Id` when it is 1-128 characters of letters, digits and `._:-`, otherwise a generated one. The frontend logs it with failed requests and sends it, with a suffix per call, to the vLLM backend. If the backend binds a specific address with `--backend-host`, start the frontend with the same `--backend-host` (or `--backend-url`).

The state, instructions and option texts are data for the model, not instructions to the server. Text that spells a tokenizer control or tag token (for example `<|im_end|>` or `</think>`) is neutralized before tokenization, so it stays plain text and cannot end the user turn; all other text is passed exactly as sent, in the prompt format used in training. A decision about adversarial text (for example a document written to steer the answer) is still a model output about that text, not a guarantee.
