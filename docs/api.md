# API reference

[Back to README](../README.md) · [Serving](serving.md) · [Native Bielik 3.2](native.md)

`POST /v1/systemone` takes a textual `state` and a mapping of named `questions`.
Answers are returned under those names. Questions about the same state can be
sent together to reuse the prefix cache. OpenAPI documentation is available at
`/docs` on the frontend.

- `choice`: choose one of the named options.
- `noul`: estimate the probability of yes, with optional descriptions for `true` and `false`.
- `score`: estimate an ordered level; the score is the expected 0-based level.
- `labels`: estimate an independent probability for each option.

## Request example

This example targets the **separate frontend** on port 8026, using the
`decision` model alias. With [adapter middleware](serving.md#single-server-middleware-adapter-packages)
or [native middleware](native.md#single-server-middleware), use the middleware
server's address/port and public served-model name (for example `bielik-base`)
instead; `decision` is not its public model alias.

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

## Answers

Answers are keyed by question name:

- `choice`: the chosen key, `probabilities` and `confidence`.
- `noul`: the probability of yes, `probabilities` (`true`, `false`) and `confidence`.
- `score`: the expected 0-based level, `probabilities` per level, `legend` and `confidence`.
- `labels`: independent `probabilities`, `selected` keys at probability >= 0.5, and `confidence`.

Adapter and merged decision packages use their labels head by default. Native
Bielik 3.2 always uses independent `noul` reads; see [Native Bielik 3.2](native.md).

## Request options

- `read_twice` defaults to `true`; the standard prompt is repeated if eligible.
- `read_twice_max_input_tokens` defaults to `4096`, measured before repetition.
  Set it to `null` to remove the threshold. Context limits still apply.
- `permutations` defaults to `1`; up to 16 option orders can be averaged.
- `questions_before_state` defaults to `false`; enabling it includes the batch's
  questions before the state in the prompt.
- `temperature` overrides saved calibration: a number applies globally; a
  mapping selects question types. Use `1` for raw probabilities.
- `facts` defaults to `false`; it appends computed calendar and number facts.
- `prompt_details` defaults to `false`; enable it for per-read prompt metadata.

See [benchmark settings](benchmarking.md#benchmarking) for the effect of each
option, including the different read-twice overflow behavior of native 3.2.

## Confidence

`confidence` uses the formulas of the reference System One implementation (Hugging Face Transformers' decision pipeline, llama.cpp), so compatible distributions use the same confidence calculation. `confidence` describes the distribution; it is not a calibrated probability of correctness (see [Calibration](benchmarking.md#calibration)).

- `choice`: (p_max - 1/K) / (1 - 1/K) for K options: 0 for a uniform distribution, 1 when one option has all the probability;
- `noul`: the same formula over its two sides, |2p - 1|;
- `score`: 1 - d / d_u, where d is the expected distance (in levels) from the most probable level and d_u the same for a uniform distribution from the middle of the scale, so a near miss counts less than a far one (floored at 0);
- `labels` (not a reference type): the product of the probabilities of each option's selected/not-selected outcome. This is an independence-based score, not a measured probability of exact-set correctness.

## Labels modes

For adapter and merged decision packages, a `labels` question takes an optional `"mode"`. `"head"` (the default) reads every option in one pass from the labels head. `"per_option"` asks one `noul` question per option, each showing the question, all options and one candidate (`Candidate: <key>: <text>` / `Does this candidate belong to the correct answer set?`), and returns the same `labels` answer with `"mode": "per_option"`.

For Bielik-One-4B-r1 (bfloat16, through this API, on the multilabel questions of the native test split) it raises exact-set accuracy by 6.0 percentage points and set F1 by 1.0 point; the effect differs by model. It costs one read per option (4.4 on average there), about 2.5 times the time per question.

Each option is a `noul` decision, so it uses the `noul` calibration temperature and the 0.5 threshold; a request `temperature` applies if it is a number or has a `noul` entry (a mapping without `noul` means temperature 1, as for any type it does not name). There is no separately fitted temperature for this mode in the adapter/merged profiles. Native 3.2 uses its own `labels` temperature, not `noul`; see [native calibration](native.md#calibration). It works with every package, including those without a labels head. `"mode"` takes only `"head"` or `"per_option"` and is rejected on other question types (it was ignored before 1.0.5).

A request may make at most 1024 decision reads in all (questions x option orders, one per option and at most two orders each for per-option labels), and the text that per-option prompts repeat is capped at 2 million characters per request; both are checked before anything is sent to vLLM.

## Computed facts

`"facts": true` (default `false`) adds deterministic preprocessing for questions that depend on exact arithmetic: dates and deadlines, day counts, amounts against thresholds. A one-pass decision cannot reliably count the days between two dates or add up a list of amounts; with `facts` the API computes such facts in code from the state and the question instructions and appends them to the state, under a one-line header, before the model reads it. Code computes, the model decides:

- every date of the state with its weekday, and its distance from today when the state gives today's date; days between the dates (also counting both days, and in working days Monday-Friday without the days off the text names when the text speaks of working days), with full months and years for long spans; clock time between dated times;
- date + duration for the durations in the instructions ("within 30 days", "10 working days", "the 14th day", "3 years"), month ranges such as `09.2019 - 08.2021` or `03.2020 - present`;
- sums of amounts with the same unit (currency, %, MB, kg, hours, ...) in one sentence, in a list and in the list lines with the same description, an amount per person when the rule speaks of one, and comparisons with the thresholds of the same unit in the instructions, as symbols (`512,40 zł > 500 zł`);
- whether the state contains a bank account number, an e-mail address or a postal code, when the instructions ask for one.

Only arithmetic and calendar facts are added, never a verdict ("the deadline is met", "the limit is exceeded"): which fact matters and how the rule counts stays the model's decision. Polish, English, German and Norwegian Bokmål are recognised (the facts are written in the request's language, detected from the text). Numbers are read in the convention of that language when their separators alone do not decide (`12.500` is 12500 in Polish, German and Norwegian text; `12,500` is 12500 in English text); with both `.` and `,` the last one is the decimal separator, and an amount of money whose separator could be either (`12,500 zł` in Polish text, `£12.500` in English text) is left out rather than guessed. The response then has `"facts": {"language": ..., "lines": [...]}`; a request without facts to compute is sent unchanged. The facts are part of the state, so with read twice they are repeated with it and count toward `read_twice_max_input_tokens` and the context; when the standard prompt with them would exceed the context, the request is read without them (`"lines": []`). Without `facts` the prompt is unchanged. The models were not trained with the block; see [Benchmarking](benchmarking.md#benchmarking) for its measured effect. The code is `bielik_one/prompting/facts.py`.

## Errors

Errors have one shape, `{"error": "<message>", "code": "<CODE>"}`, with `detail` (the field locations) on a 422. The codes:

- `INVALID_REQUEST` (422): the request must be changed, including a body that is not valid JSON;
- `PAYLOAD_TOO_LARGE` (413);
- `NOT_FOUND` (404) and `METHOD_NOT_ALLOWED` (405);
- `BACKEND_UNAVAILABLE` (502; 503 on `/health`) and `TIMEOUT` (504), when vLLM does not answer or not in time;
- `INTERNAL_ERROR` (500).

Overloaded bounded native requests return 503 with `Retry-After: 1`.

The chat routes answer errors in the OpenAI format, `{"error": {"message", "type", "param", "code"}}`, as vLLM itself does there. `bielik-one frontend ... --error-format openai` uses that format on every route except `/health`, which keeps `{"status", "error", "code"}` for load-balancer probes.

## Chat and completions

Ordinary chat and completions go to `/v1/chat/completions` and `/v1/completions` on the same port, with the package's chat model alias (listed by `/v1/models`, e.g. `bielik-base`, `qwen-base` or `qwen-general`; requests without a model get it); the `decision` model is rejected there. One such request may ask for at most 16 answers per prompt (`n`, or `best_of`, which is also the beam width for beam search) and at most 131,072 output tokens in all, counted as prompts x answers x tokens per answer. Tokens per answer are read from the field the route uses, as vLLM does (`max_tokens` for completions and messages, `max_completion_tokens` and then `max_tokens` for chat, `max_output_tokens` for responses); an answer without that limit counts as the full 32,768-token context, and on other routes the costliest reading counts. A request over either limit gets 400 naming the field. `--max-generation-n` and `--max-generation-tokens` on `bielik-one frontend` change them (`none` lifts a limit), and `SYSTEMONE_MAX_GENERATION_N` / `SYSTEMONE_MAX_GENERATION_TOKENS` do the same for the vLLM middleware. Interactive docs: `/docs`.
