# bielik-one

Typed decision API and launcher for the Bielik One decision models served with vLLM: Bielik-One-11B-v3.0-r1, Bielik-One-11B-v3.1-r1, Bielik-One-7B-v3.0-r1 and Bielik-One-4B-r1, each as an adapter package (`-Adapter`) and as a merged export.

The API answers four question types about a textual state — `choice`, `noul` (binary), `score` and `labels` — with probabilities, in one forward pass per question. **`labels` is native multi-label classification:** a dedicated labels head scores all listed options together and gives each an independent probability; it is not emulated as a series of `noul` questions, one per option (that protocol is available on request with `"mode": "per_option"`, see below). Model weights live in the Hugging Face repositories; this repository holds the code that serves them.

## Install

Python 3.12 and an NVIDIA GPU. In a fresh environment:

```bash
pip install "bielik-one[vllm] @ git+https://github.com/speakleash/bielik-one"
# only for adapter packages (*-Adapter): the decision-heads plugin
pip install --no-deps "systemone-vllm-heads @ git+https://github.com/speakleash/bielik-one#subdirectory=vllm-heads-plugin"
```

## Run

Start the backend and, in a second terminal, the API, with the model's Hugging Face repo id:

```bash
bielik-one backend speakleash/Bielik-One-11B-v3.0-r1
bielik-one frontend speakleash/Bielik-One-11B-v3.0-r1
```

The package is downloaded to the Hugging Face cache on first use (`--revision` pins a commit). A local directory, for example from `hf download ... --local-dir model`, works the same: `bielik-one backend model`. `bielik-one backend <package> --dry-run` verifies the published checksums and prints the exact vLLM command. The launcher detects the package type and uses the profile validated for it:

| Package | Backend | API |
|---|---|---|
| `Bielik-One-11B-v3.0-r1-Adapter`, `Bielik-One-11B-v3.1-r1-Adapter`, `Bielik-One-7B-v3.0-r1-Adapter` | vLLM + LoRA + heads plugin | `http://127.0.0.1:8026` |
| `Bielik-One-11B-v3.0-r1`, `Bielik-One-11B-v3.1-r1`, `Bielik-One-7B-v3.0-r1` | stock vLLM | `http://127.0.0.1:8026` |
| `Bielik-One-4B-r1-Adapter` | vLLM + LoRA + heads plugin | `http://127.0.0.1:8012` |
| `Bielik-One-4B-r1` | stock vLLM | `http://127.0.0.1:8012` |

**Adapter packages are a small add-on.** They share the unchanged base weights, so if you already serve the base model (for example `speakleash/Bielik-11B-v3.0-Instruct`) with vLLM, adding a decision adapter costs only its LoRA weights and heads: about 1.0 GB for the 11B models, 1.4 GB for the 7B model and 0.7 GB for the 4B model. The same engine keeps serving ordinary chat with the original base model. Merged packages need their own full copy of the weights.

The backend runs in **bfloat16** by default, the precision of all reported results and of the calibration. An 11B model then needs about 30 GB of GPU memory (weights plus the configured KV cache), a 7B model about 22 GB and the 4B model about 12 GB. On smaller GPUs add `--fp8` (on-load FP8 weight quantization, BF16 language-model head): it roughly halves weight memory and raises throughput (1.4x measured for the 4B model), but changes 2-18% of decisions near the decision boundary depending on model and question type, and the calibration was not fitted for it. The backend uses CUDA graphs, which speed up chat generation and leave decisions unchanged; `--eager` turns them off for a shorter start or if graphs cause problems on a GPU. For many concurrent users, `--max-num-seqs N` (pass it to both the backend and the frontend) and `--kv-cache-gb GB` raise the validated profile, for example `--max-num-seqs 32 --kv-cache-gb 4` for `Bielik-One-4B-r1` (about 14 GB of GPU memory instead of 10.5 GB). For throughput, send all questions about one state in one request: the state is read once and the other questions reuse the prefix cache. Use `--port` / `--backend-url` to move ports.

```bash
curl -s localhost:8026/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "decision",
  "state": "Invoice 12/2026 for EUR 4,300 has not been paid. The payment deadline passed yesterday. The customer has paid late twice this year.",
  "questions": {
    "status": {
      "type": "choice",
      "instructions": "What is the status of the invoice?",
      "criteria": {
        "paid": "Paid",
        "overdue": "Unpaid and past the deadline",
        "not_due": "Unpaid, deadline not reached"
      }
    },
    "late": {
      "type": "noul",
      "instructions": "Has the payment deadline passed?",
      "criteria": {
        "true": "Yes",
        "false": "No"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is collection?",
      "criteria": [
        "none",
        "low",
        "medium",
        "high",
        "critical"
      ]
    },
    "actions": {
      "type": "labels",
      "instructions": "Which actions are appropriate now?",
      "criteria": {
        "reminder": "Send a payment reminder",
        "interest": "Charge late-payment interest",
        "cancel": "Cancel the invoice"
      }
    }
  }
}'
```

Answers, keyed by question name: `choice` gives the chosen key, `probabilities` and `confidence`; `noul` the probability of yes, `probabilities` (`true`, `false`) and `confidence`; `score` the expected level (0-based over the listed levels), `probabilities` per level, `legend` and `confidence`; `labels` an independent probability per option and the `selected` keys (probability >= 0.5), all from the native labels head, and `confidence`.

`confidence` uses the formulas of the reference System One implementation (Hugging Face Transformers' decision pipeline, llama.cpp), so the same request gets the same `confidence` on any System One server. It says how concentrated the distribution is; it is not the probability that the answer is right, which is the chosen answer's entry in `probabilities` (calibrated, see [Calibration](#calibration)):
- `choice`: (p_max - 1/K) / (1 - 1/K) for K options: 0 for a uniform distribution, 1 when one option has all the probability;
- `noul`: the same formula over its two sides, |2p - 1|;
- `score`: 1 - d / d_u, where d is the expected distance (in levels) from the most probable level and d_u the same for a uniform distribution from the middle of the scale, so a near miss counts less than a far one (floored at 0);
- `labels` (not a reference type): the probability that the selected set is exactly right, the product over the options of the probability of each option's decision.

A `labels` question takes an optional `"mode"`. `"head"` (the default) reads every option in one pass from the labels head. `"per_option"` asks one `noul` question per option, each showing the question, all options and one candidate (`Candidate: <key>: <text>` / `Does this candidate belong to the correct answer set?`), and returns the same `labels` answer with `"mode": "per_option"`. For Bielik-One-4B-r1 (bfloat16, through this API, on the multilabel questions of the native test split) it raises exact-set accuracy by 6.0 percentage points and set F1 by 1.0 point; the effect differs by model. It costs one read per option (4.4 on average there), about 2.5 times the time per question. Each option is a `noul` decision, so it uses the `noul` calibration temperature and the 0.5 threshold; a request `temperature` applies if it is a number or has a `noul` entry (a mapping without `noul` means temperature 1, as for any type it does not name). There is no separately fitted temperature for this mode. It works with every package, including those without a labels head. `"mode"` takes only `"head"` or `"per_option"` and is rejected on other question types (it was ignored before 1.0.5). A request may make at most 1024 decision reads in all (questions x option orders, one per option and at most two orders each for per-option labels), and the text that per-option prompts repeat is capped at 2 million characters per request; both are checked before anything is sent to vLLM.

Errors have one shape, `{"error": "<message>", "code": "<CODE>"}`, with `detail` (the field locations) on a 422. The codes:
- `INVALID_REQUEST` (422): the request must be changed, including a body that is not valid JSON;
- `PAYLOAD_TOO_LARGE` (413);
- `NOT_FOUND` (404) and `METHOD_NOT_ALLOWED` (405);
- `BACKEND_UNAVAILABLE` (502; 503 on `/health`) and `TIMEOUT` (504), when vLLM does not answer or not in time;
- `INTERNAL_ERROR` (500).

The chat routes answer errors in the OpenAI format, `{"error": {"message", "type", "param", "code"}}`, as vLLM itself does there. `bielik-one frontend ... --error-format openai` uses that format on every route except `/health`, which keeps `{"status", "error", "code"}` for load-balancer probes.

Ordinary chat and completions go to `/v1/chat/completions` and `/v1/completions` on the same port, with the package's chat model alias (listed by `/v1/models`, e.g. `bielik-base`, `qwen-base` or `qwen-general`; requests without a model get it); the `decision` model is rejected there. Interactive docs: `/docs`.

## Running vLLM yourself

`bielik-one backend` only verifies the package and starts an ordinary `vllm serve` with a validated set of flags; the decisions are computed by `bielik-one frontend`. To run vLLM under your own process manager or container, print the exact command and start it yourself:

```bash
bielik-one backend <package> --shell     # one line: env VAR=... python -m vllm.entrypoints.cli.main serve ...
bielik-one backend <package> --dry-run   # the same as JSON: "command" and "env"
bielik-one frontend <package> --backend-url http://127.0.0.1:<port>
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

## Deployment and untrusted text

Both processes bind to `127.0.0.1` by default. `--host` moves only the frontend (typed API and chat proxy); the raw vLLM backend stays on loopback unless you pass `--backend-host`, because it accepts internal decision flags. The API has no authentication: to expose it, put it behind a reverse proxy that authenticates clients. Requests are limited to 64 questions, 16 permutations, 400,000 characters of state and 400,000 characters of questions. The frontend refuses a body larger than 16 MiB with HTTP 413 before reading it whole (`--max-body-mb` on `bielik-one frontend` changes it); text that is not valid Unicode (a lone UTF-16 surrogate such as `"\udead"`) is refused with 422. Every response carries an `x-request-id` header: the client's `X-Request-Id` when it is 1-128 characters of letters, digits and `._:-`, otherwise a generated one. The frontend logs it with failed requests and sends it, with a suffix per call, to the vLLM backend. If the backend binds a specific address with `--backend-host`, start the frontend with the same `--backend-host` (or `--backend-url`).

The state, instructions and option texts are data for the model, not instructions to the server. Text that spells a tokenizer control or tag token (for example `<|im_end|>` or `</think>`) is neutralized before tokenization, so it stays plain text and cannot end the user turn; all other text is passed exactly as sent, in the prompt format used in training. A decision about adversarial text (for example a document written to steer the answer) is still a model output about that text, not a guarantee.

## Calibration

Each model repository ships `calibration.json`: one temperature per question type, fitted on validation data only. The API applies it by default in every request mode: any number of option orders, `read_twice` and `questions_before_state`. The temperatures were fitted for one option order on the standard prompt, and they carry over: for read-twice prompts and FP8 serving, temperatures refitted for those settings lowered test NLL by at most 0.012 against the shipped ones (Bielik-One-11B-v3.0-r1, Bielik-One-7B-v3.0-r1 and Bielik-One-4B-r1 for read-twice; these and Bielik-One-11B-v3.1-r1 for FP8). With 2 or 4 option orders and with `questions_before_state` the shipped temperatures gave a lower NLL than raw probabilities for every categorical type in a check on native validation (Bielik-One-4B-r1); a temperature fitted for `questions_before_state` would be somewhat higher. Calibration never changes argmax or 0.5-threshold decisions; the `score` field is the expected level and does change. Raw probabilities: start the API with `--no-calibration`, or send `"temperature": 1`. `/health` reports whether calibration is active. The temperatures were fitted on bfloat16 outputs with the standard prompt.

## Benchmarking

The request options below change the numbers you get, so a result is only meaningful together with the settings it was measured with. Report them, and when you compare with another decision model, match its settings instead of comparing defaults: if that model reads the input twice or averages two option orders by default, run these models the same way too.

- **Reference profile.** bfloat16, one option order (`permutations: 1`), `read_twice: false`, `questions_before_state: false`, labels in `"mode": "head"`. This is the setting of the results on the model cards.
- **Probabilities.** Calibration does not change any decision, so accuracy is the same with and without it. For probability metrics (NLL, Brier, ECE, KL) say which you used: the calibrated default, or raw with `"temperature": 1`. Calibration applies in every mode (see [Calibration](#calibration)), so raw probabilities need `"temperature": 1` explicitly.
- **Labels.** `"mode": "head"` reads all options in one pass. `"mode": "per_option"` asks one yes/no question per option, which is how decision models without a multi-label head answer such questions, so use it when comparing with them. For Bielik-One-4B-r1 it raises exact-set accuracy by 6.0 percentage points on the native test split, at one read per option.
- **Read twice (default).** `"read_twice": true` repeats the input when the complete standard prompt has at most `read_twice_max_input_tokens` tokens (default **4096**, before repetition). Longer prompts use the standard input. Set `"read_twice": false` to disable it, or `"read_twice_max_input_tokens": null` to remove the length threshold. The exact repeated prompt plus one output token must always fit the server context; otherwise the standard prompt is used. If the standard prompt also exceeds context, the request fails without truncation. This changes the behavior of requests that omit `read_twice`; reference benchmark requests must explicitly send `false`. The 4096 threshold is a provisional cost/accuracy tradeoff from 4B native validation, not a validated optimum for every model. Earlier unrestricted read-twice measurements raised single-choice accuracy on the native test split by 3.5 percentage points (Bielik-One-11B-v3.0-r1), 3.3 (Bielik-One-7B-v3.0-r1) and 1.5 (Bielik-One-4B-r1), at about twice the prompt length; it was not measured for Bielik-One-11B-v3.1-r1.
- **Option orders.** `"permutations": N` (up to 16) averages N orders of the options, which reduces position bias; it costs N reads per question.
- **Repeatability.** Batching under concurrent load changes floating-point order, so a few decisions near the boundary can differ between identical requests (98-100% agreement in our A100 measurements). For repeatable runs keep the `--max-num-seqs` profile fixed and report it. `--fp8` changes 2-18% of decisions depending on model and type; benchmark in bfloat16 unless FP8 is what you deploy.
- **Long inputs.** Nothing is truncated: a prompt over the 32,768-token context is refused with 422. Count it as an error instead of dropping the question, so every system is scored on the same denominator.
- **Request layout.** Questions about the same state can go in one request; the state is then read once. This changes speed, not answers.
- **External harnesses.** `choice`, `noul` and `score` follow the TypeSafe System One schema, so a harness that calls `/v1/systemone` (for example a public Typed Decisions scorer) works against this API unchanged; `labels` and the options above are extensions.

## Contents

`bielik_one/launch.py` (launcher), `bielik_one/api/` (typed API frontend over a vLLM backend), `bielik_one/prompting/` (prompt rendering), `vllm-heads-plugin/` (`systemone-vllm-heads`, decision heads inside vLLM).

## Tests

```bash
pip install -e ".[test]" && pytest -q
```

The tests use synthetic backends and need no GPU.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to report bugs, run the tests and check changes that can affect model outputs; report security issues privately through the repository's Security tab.

## License

Apache License 2.0, see `LICENSE`. Model weights are distributed separately under their own licenses.
