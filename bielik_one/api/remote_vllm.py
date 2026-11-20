"""SystemOne HTTP API backed by an independent vLLM server; no GPU weights here."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import contextvars
from dataclasses import dataclass
import json
import math
from pathlib import Path

import httpx
from fastapi import Request

from bielik_one.api.server import (
    CONTEXT_LIMIT,
    DEFAULT_READ_TWICE_MAX_INPUT_TOKENS,
    LetterEngine,
    RequestError,
    SystemOneIn,
    build_app,
    option_orders,
)
from bielik_one.api.errors import (
    BACKEND_UNAVAILABLE,
    INVALID_REQUEST,
    TIMEOUT,
    error_response,
)
from bielik_one.api.http_guard import add_guard_arguments, backend_request_id, guarded
from bielik_one.api.reasoning import ReasoningError
from bielik_one.prompting.inference import inference_ids
from bielik_one.prompting.letter import answer_token_ids

# vLLM 0.32 caps logprob_token_ids at 128 per request (vllm.sampling_params.MAX_LOGPROB_TOKEN_IDS).
MAX_LOGPROB_TOKEN_IDS = 128

# Request-local collection: concurrent requests must never share inference metadata.
_prompt_variants = contextvars.ContextVar("prompt_variants", default=None)

# calibration.json uses benchmark names; the TypeSafe API calls binary questions "noul".
CALIBRATION_KINDS = {
    "choice": "choice",
    "binary": "noul",
    "score": "score",
    "labels": "labels",
}


log = logging.getLogger(__name__)


def load_calibration(path: Path, model: str | None = None) -> dict:
    """Validation-fitted temperatures as the server's `temperature_by_orders` table.

    They were fitted for one option order on the standard prompt, so only table "1" is set; it applies to every
    request mode (more option orders, read_twice, questions_before_state; see LetterEngine._evaluate_once). Only the
    reasoning second pass stays at temperature 1, and auto-reasoning gates on raw confidence.
    A request `temperature` still wins. `model` (the metadata source_repo) must match the file.
    """
    data = json.loads(Path(path).read_text())
    if model is not None and data.get("model") != model:
        raise ValueError(
            f"calibration.json is for {data.get('model')!r}, not for {model!r}"
        )
    temperatures = data.get("temperatures")
    if not isinstance(temperatures, dict) or set(temperatures) != set(
        CALIBRATION_KINDS
    ):
        raise ValueError(
            "calibration.json must give temperatures for choice, binary, score and labels"
        )
    table = {}
    for name, kind in CALIBRATION_KINDS.items():
        value = temperatures[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"calibration temperature for {name} must be a positive number"
            )
        table[kind] = float(value)
    return {"1": table}


def resolve_calibration(
    metadata_path: Path,
    calibration: Path | None,
    disabled: bool,
    model: str | None = None,
) -> tuple[dict, str | None]:
    """Explicit file, else calibration.json next to the metadata, else none. Returns (table, source)."""
    if disabled:
        log.warning("calibration disabled: probabilities are raw (temperature 1)")
        return {}, None
    if calibration is None:
        default = package_dir(metadata_path, "calibration.json") / "calibration.json"
        if not default.is_file():
            log.warning(
                "no calibration.json next to %s: probabilities are raw (temperature 1)",
                metadata_path,
            )
            return {}, None
        calibration = default
    return load_calibration(calibration, model), str(calibration)


def package_dir(metadata_path, wanted: str) -> Path:
    """The directory holding `wanted` next to the metadata file.

    The link's own directory comes first (in the HF cache, metadata is a symlink into blobs/ and the package files
    sit next to the link); otherwise the link chain is followed hop by hop.
    """
    path = first = Path(metadata_path).absolute()
    for _ in range(40):  # follow the link chain one hop at a time
        if (path.parent / wanted).exists():
            return path.parent
        if not path.is_symlink():
            break
        path = path.parent / path.readlink()
    return first.parent


class BackendReadError(RuntimeError):
    pass


class RemoteVllmEngine(LetterEngine):
    def __init__(
        self,
        metadata: dict,
        backend_url: str,
        *,
        model_name="decision",
        backend_model="decision",
        max_model_len=32768,
        tokenizer=None,
        client=None,
        workers=8,
        calibration=None,
        calibration_source=None,
    ):
        from transformers import AutoTokenizer

        self.tok = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                metadata["base_snapshot"], local_files_only=True
            )
        )
        # A package may pin the chat template used in training; the base repository's template can differ
        # (e.g. the public Bielik 3.1 base adds a dated system preamble). Decision prompts must use the pinned one.
        self.chat_template = metadata.get("chat_template_path")
        if self.chat_template:
            self.tok.chat_template = Path(self.chat_template).read_text(
                encoding="utf-8"
            )
        self.backend_url = backend_url.rstrip("/")
        self.backend_model = backend_model
        self.model_name = model_name
        self.aliases = {
            model_name,
            backend_model,
            metadata.get("source_repo", model_name),
        }
        self.capacity = int(metadata["choice_capacity"])
        self.labels_capacity = int(metadata["labels_capacity"])
        self.bias = [float(b) for b in metadata["bias"]]
        self.parking = {
            "option_ids": metadata["label_ids"],
            "anchor_id": metadata["anchor_id"],
        }
        assert self.labels_capacity == len(self.bias) == len(self.parking["option_ids"])
        assert (
            len(set(self.parking["option_ids"] + [self.parking["anchor_id"]]))
            == self.labels_capacity + 1
        )
        self.labels_threshold = float(metadata["threshold"])
        if calibration and self.labels_threshold != 0.5:
            raise ValueError(
                "temperature calibration preserves labels decisions only at threshold 0.5"
            )
        self.max_model_len = max_model_len
        self.semantic = True
        # Routed packages: these question types run on the unchanged base model, without
        # the adapter; the backend's logits processor applies the decision heads to rows flagged in vllm_xargs.
        self.base_row_types = set(metadata.get("base_row_types") or ())
        self.base_alias = metadata.get("base_alias")
        if self.base_row_types and not self.base_alias:
            raise ValueError("base_row_types requires base_alias in the metadata")
        if not self.base_row_types <= {"choice", "noul", "score", "labels"}:
            raise ValueError("unknown base_row_types")
        head_ids = set(self.parking["option_ids"]) | {self.parking["anchor_id"]}
        for values in (metadata.get("categorical_ids") or {}).values():
            head_ids.update(values)
        self.canary_id = next(i for i in range(len(head_ids) + 1) if i not in head_ids)
        # Default probabilities use these validation-fitted temperatures; {} means raw (T=1).
        self.temperatures = calibration or {}
        self.calibration_source = calibration_source if self.temperatures else None
        self.reasoners = {}
        self.client = client or httpx.Client(
            timeout=180,
            trust_env=False,  # loopback backend: ignore HTTP_PROXY
            limits=httpx.Limits(
                max_connections=workers, max_keepalive_connections=workers
            ),
        )
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.client.close()

    def _chunk_width(self, on_base: bool) -> int:
        """Option ids per backend read: vLLM accepts at most MAX_LOGPROB_TOKEN_IDS ids per request, and every
        read also carries the anchor and, on flagged base rows, the canary."""
        return MAX_LOGPROB_TOKEN_IDS - 1 - (1 if on_base else 0)

    def _read(self, prompt: list[int], output_ids: list[int], kind: str):
        """Every chunk includes a common anchor, cancelling its softmax normalizer."""
        anchor = int(self.parking["anchor_id"])
        on_base = kind in self.base_row_types
        logits = {}
        width = self._chunk_width(on_base)
        for offset in range(0, len(output_ids), width):
            chunk = output_ids[offset : offset + width]
            requested = chunk + [anchor]
            # Flagged base rows must come back as head-only logits: a non-head canary must be -inf. If the backend
            # was started without the logits processor it would be finite, and the read fails instead of
            # returning base-model vocabulary probabilities.
            canary = [self.canary_id] if on_base else []
            payload = {
                "model": self.base_alias if on_base else self.backend_model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
                "min_p": 0,
                "repetition_penalty": 1,
                "logprobs": 1,
                "logprob_token_ids": requested + canary,
                "return_tokens_as_token_ids": True,
                "seed": 0,
            }
            if on_base:
                payload["vllm_xargs"] = {"systemone_head": "base"}
            try:
                rid = backend_request_id()
                response = self.client.post(
                    self.backend_url + "/v1/completions",
                    json=payload,
                    headers={"X-Request-Id": rid} if rid else None,
                )
                response.raise_for_status()
                result = response.json()
                values = result["choices"][0]["logprobs"]["top_logprobs"][0]
                table = {
                    int(k.removeprefix("token_id:")): float(v)
                    for k, v in values.items()
                }
                if any(
                    i not in table or not math.isfinite(table[i]) for i in requested
                ):
                    raise ValueError("missing or non-finite requested logprobs")
                if canary and (canary[0] not in table or table[canary[0]] > -9990):
                    raise ValueError(
                        "backend did not apply decision heads to a flagged base row (logits processor missing?)"
                    )
                if any(table[i] <= -9990 for i in requested):
                    raise ValueError("invalid or clamped requested logprob")
                if result["usage"]["prompt_tokens"] != len(prompt):
                    raise ValueError("backend changed prompt length")
            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise BackendReadError(
                    "vLLM did not return a complete decision read"
                ) from exc
            for token_id in chunk:
                logits[token_id] = table[token_id] - table[anchor]
        if kind == "labels":
            # The inherited reader applies sigmoid(logit - anchor); add bias once.
            return {
                **{
                    token: logits[token] + self.bias[i]
                    for i, token in enumerate(output_ids)
                },
                anchor: 0.0,
            }, len(prompt)
        peak = max(logits.values())
        normalizer = peak + math.log(sum(math.exp(v - peak) for v in logits.values()))
        return {token: value - normalizer for token, value in logits.items()}, len(
            prompt
        )

    def distributions(
        self,
        orders,
        *,
        questions=(),
        first_rows=(),
        questions_before_state=False,
        read_twice=False,
        read_twice_max_input_tokens=DEFAULT_READ_TWICE_MAX_INPUT_TOKENS,
        analyses=None,
    ):
        if first_rows and len(first_rows) != len(orders):
            raise ValueError("Expected one original row per decision read")
        standard = []
        groups = {}
        for index, row in enumerate(orders):
            if row.question.type == "labels" and len(row.options) > self.labels_capacity:
                raise RequestError(
                    ["questions", row.id, "criteria"],
                    f"labels supports at most {self.labels_capacity} options",
                )
            ids = inference_ids(
                self.tok, row, self.semantic, questions=questions,
                questions_before_state=questions_before_state, read_twice=False,
                analysis=(analyses or {}).get(row.id, ""),
            )
            if len(ids) + 1 > self.max_model_len:
                raise RequestError(["questions", row.id], CONTEXT_LIMIT)
            standard.append(ids)
            groups.setdefault(row.id, []).append(index)

        selected = list(standard)
        decisions = {}
        for name, indices in groups.items():
            reason = "disabled"
            if read_twice:
                longest = max(len(standard[i]) for i in indices)
                if read_twice_max_input_tokens is not None and longest > read_twice_max_input_tokens:
                    reason = "input_threshold"
                else:
                    repeated = [inference_ids(
                        self.tok, orders[i], self.semantic,
                        first_row=first_rows[i] if first_rows else orders[i],
                        questions=questions, questions_before_state=questions_before_state,
                        read_twice=True, analysis=(analyses or {}).get(name, ""),
                    ) for i in indices]
                    if all(len(ids) + 1 <= self.max_model_len for ids in repeated):
                        reason = "applied"
                        for i, ids in zip(indices, repeated, strict=True):
                            selected[i] = ids
                    else:
                        reason = "context_limit"
            decisions[name] = reason

        tasks = []
        variants = _prompt_variants.get()
        for row, base, ids in zip(orders, standard, selected, strict=True):
            kind = row.question.type
            reason = decisions[row.id]
            if variants is not None:
                variants.append((row.id, {
                    "read_twice": reason == "applied",
                    "reason": reason,
                    "standard_input_tokens": len(base),
                    "input_tokens": len(ids),
                }))
            outputs = (
                self.parking["option_ids"][:len(row.options)]
                if kind == "labels" else answer_token_ids(self.tok, row, self.semantic)
            )
            tasks.append((ids, list(outputs), kind))
        # Each read runs in a copy of the request's context, so it sees the request id.
        futures = [
            self.pool.submit(contextvars.copy_context().run, self._read, *task)
            for task in tasks
        ]
        results = [future.result() for future in futures]
        return [table for table, _ in results], sum(tokens for _, tokens in results)

    def _evaluate_once(self, body, **kwargs):
        variants = []
        token = _prompt_variants.set(variants)
        try:
            result = super()._evaluate_once(body, **kwargs)
        finally:
            _prompt_variants.reset(token)
        for name, answer in result["answers"].items():
            reads = [record for question_id, record in variants if question_id == name]
            if reads:
                applied = all(record["read_twice"] for record in reads)
                answer["prompt"] = {
                    "requested_read_twice": body.read_twice,
                    "variant": "read_twice" if applied else "standard",
                    "reason": reads[0]["reason"],
                    "read_twice_max_input_tokens": body.read_twice_max_input_tokens,
                    "max_context_tokens": self.max_model_len,
                    "read_twice_reads": sum(record["read_twice"] for record in reads),
                    "standard_reads": sum(not record["read_twice"] for record in reads),
                }
                if body.prompt_details:
                    answer["prompt"]["reads"] = reads
        calls = 0
        seed = 0 if body.seed is None else body.seed
        for question in body.questions.values():
            if question.type == "labels" and question.mode == "per_option":
                # One noul read per option, in the noul option orders.
                width = self._chunk_width("noul" in self.base_row_types)
                calls += len(question.criteria) * -(-2 // width) * len(
                    option_orders(2, body.permutations, seed)
                )
                continue
            n = 2 if question.type == "noul" else len(question.criteria)
            width = self._chunk_width(question.type in self.base_row_types)
            calls += -(-n // width) * len(option_orders(n, body.permutations, seed))
        result["usage"]["output_tokens"] = calls
        return result

    def evaluate(self, body: SystemOneIn):
        if body.model not in self.aliases:
            raise RequestError(["model"], f"use model {self.model_name!r}")
        return super().evaluate(body)


# What one public generation request (chat, completions, responses, messages) may ask vLLM to produce. vLLM itself
# accepts n up to 16384 and up to 1024 prompts per completions request, each answer running to the end of the context
# when max_tokens is not set.
DEFAULT_MAX_GENERATION_N = 16
DEFAULT_MAX_GENERATION_TOKENS = 131_072  # four full answers at the 32k context of every launcher profile
CONTEXT_TOKENS = 32768  # --max-model-len of the launcher profiles: how long an answer without max_tokens can run


@dataclass(frozen=True)
class GenerationLimits:
    """Per-request budget for public generation. None lifts a limit."""

    max_n: int | None = DEFAULT_MAX_GENERATION_N
    max_tokens: int | None = DEFAULT_MAX_GENERATION_TOKENS
    context_tokens: int = CONTEXT_TOKENS


def _count(value) -> int | None:
    """A count or token limit as vLLM's request models would read it. Pydantic coerces "8", 8.0 and true to ints, so
    those count too; fractions are rounded up, so nothing is counted lower than vLLM could use it."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return math.ceil(value)
    return None


# Which limit field each route reads, as vLLM 0.32 does: a reading is the fields tried in order, the first one set
# wins, and an answer with none of them set runs to the end of the context. Chat prefers max_completion_tokens.
_CHAT = ("max_completion_tokens", "max_tokens")
ROUTE_READINGS = {
    "/v1/completions": (("max_tokens",),),
    "/v1/chat/completions": (_CHAT,),
    "/v1/chat/completions/batch": (_CHAT,),
    "/v1/responses": (("max_output_tokens",),),
    "/v1/messages": (("max_tokens",),),
}
# Any other route (for example /invocations, which picks the handler from the body): the costliest of the readings.
ANY_READING = (_CHAT, ("max_tokens",), ("max_output_tokens",))


# vLLM routes that never generate output tokens (tokenization, rendering, scoring with max_tokens=1, embeddings,
# pooling, reranking, status and admin calls): the budget does not apply to them.
NON_GENERATION_ROUTES = (
    "/tokenize", "/detokenize", "/v1/messages/count_tokens", "/generative_scoring",
    "/v1/chat/completions/render", "/v1/chat/completions/derender", "/v1/completions/render",
    "/v1/completions/derender", "/v1/messages/render", "/v1/responses/render",
    "/v1/embeddings", "/v2/embed", "/pooling", "/classify", "/score", "/v1/score", "/rerank", "/v1/rerank", "/v2/rerank",
    "/ping", "/is_scaling_elastic_ep",
)
# On a route the filter knows nothing about, a body that asks for none of these is not a generation request.
GENERATION_FIELDS = ("prompt", "messages", "input", "token_ids", "sampling_params", "n", "best_of", *(
    "max_completion_tokens", "max_tokens", "max_output_tokens"))


def _on(route: str, path: str) -> bool:
    """route is path, possibly behind a root path."""
    return route == path or route.endswith(path) and route[: -len(path)].startswith("/")


def generates(route: str | None, body: dict) -> bool:
    """Whether the generation budget applies to this request."""
    if route and any(_on(route, path) for path in NON_GENERATION_ROUTES):
        return False
    if route and any(_on(route, path) for path in ROUTE_READINGS) or route and _on(route, "/inference/v1/generate"):
        return True
    return any(field in body for field in GENERATION_FIELDS)


def _readings(route: str | None) -> tuple[tuple[str, ...], ...]:
    if route:
        for path, readings in ROUTE_READINGS.items():
            if _on(route, path):
                return readings
    return ANY_READING


def check_generation_budget(
    body: dict, limits: GenerationLimits, route: str | None = None, loc: tuple[str, ...] = ()
) -> None:
    """Refuse a request that asks for more answers, or more output tokens in all, than the limits allow.

    The cost counted is prompts x answers per prompt x tokens per answer. Prompts are the longer of a completions
    prompt list and a batch chat's list of conversations (vLLM ignores the field its route does not use, so neither
    may lower the count). n, or best_of (also the beam width for beam search), multiplies each. Tokens per answer come
    from the field the route reads (`route` is the request path), and are the full context when that field is not
    set; for a route not listed, the costliest reading counts. Token-in requests carry these fields in
    sampling_params, which is checked the same way.
    """
    n = _count(body.get("n")) or 1
    best_of = _count(body.get("best_of")) or 0
    width, width_field = (best_of, "best_of") if best_of > n else (n, "n")
    if limits.max_n is not None and width > limits.max_n:
        raise RequestError([*loc, width_field], f"{width_field} is {width}; this server allows at most {limits.max_n}")
    prompts = 1
    prompt = body.get("prompt")
    if isinstance(prompt, list) and prompt and not all(isinstance(token, int) for token in prompt):
        prompts = len(prompt)  # several prompts (text or token lists), not one prompt given as token ids
    messages = body.get("messages")
    if isinstance(messages, list) and messages and all(isinstance(m, list) for m in messages):
        prompts = max(prompts, len(messages))  # /v1/chat/completions/batch: one answer per conversation
    costs = []  # (tokens per answer, field named in an error, whether it was unset)
    for reading in (("max_tokens",),) if loc else _readings(route):
        found = next(((v, name) for name in reading if (v := _count(body.get(name))) is not None and v >= 1), None)
        if found is None:
            costs.append((limits.context_tokens, reading[-1], True))
        else:
            costs.append((min(found[0], limits.context_tokens), found[1], False))
    per_answer, field, unset = max(costs)
    total = prompts * width * per_answer
    if limits.max_tokens is not None and total > limits.max_tokens:
        unit = f"{per_answer} tokens each" + (f" ({field} not set)" if unset else "")
        raise RequestError(
            [*loc, field],
            f"this request asks for up to {total} output tokens ({prompts} prompt(s) x {width} answer(s) x {unit}); "
            f"this server allows at most {limits.max_tokens} per request. Set {field} or ask for fewer answers",
        )
    if not loc and isinstance(body.get("sampling_params"), dict):
        check_generation_budget(body["sampling_params"], limits, route, ("sampling_params",))


def public_request_body(
    raw: bytes,
    decision_models=frozenset(),
    chat_model: str | None = None,
    limits: GenerationLimits = GenerationLimits(),
    route: str | None = None,
) -> bytes:
    """Ordinary chat/completion requests from clients: drop internal decision flags; reject prompt logprobs and
    requests over the generation budget.

    vllm_xargs could switch rows to decision heads; prompt logprobs and echo+logprobs are not supported by the
    decision-head backend. The decision model names are rejected: they answer typed questions at /v1/systemone,
    and text generated through the decision adapter is not a supported output. Bodies that are not JSON objects
    pass through for vLLM to reject.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(body, dict):
        return raw
    changed = False
    model = body.get("model")
    if not model and chat_model:
        # vLLM serves a request without a model (or with "") under its default served name
        body["model"] = model = chat_model
        changed = True
    if isinstance(model, str) and model in decision_models:
        hint = f"; use model {chat_model!r} for text generation" if chat_model else ""
        raise RequestError(
            ["model"],
            f"model {model!r} answers typed decisions at /v1/systemone only{hint}",
        )
    if body.get("prompt_logprobs") is not None or (
        body.get("echo")
        and (body.get("logprobs") is not None or body.get("top_logprobs") is not None)
    ):
        raise RequestError(
            ["prompt_logprobs"], "prompt logprobs are not supported by this server"
        )
    if generates(route, body):
        check_generation_budget(body, limits, route)
    if "vllm_xargs" in body:
        body.pop("vllm_xargs")
        changed = True
    return json.dumps(body).encode() if changed else raw


def install_backend_errors(app, error_format: str = "default"):
    """Answer failed backend reads and reasoning calls of a decision app with typed 502/504/422 errors."""

    def failure(request, status, code, message):
        return error_response(status, code, message, path=request.url.path, fmt=error_format)

    @app.exception_handler(ReasoningError)
    async def reasoning_error(request, exc):
        log.warning("reasoning failed: %s", exc)
        if exc.status == 422:
            return failure(request, 422, INVALID_REQUEST, str(exc))
        if exc.status == 504:
            return failure(request, 504, TIMEOUT, "reasoning endpoint did not answer in time")
        return failure(request, 502, BACKEND_UNAVAILABLE, "reasoning endpoint failed")

    @app.exception_handler(BackendReadError)
    async def backend_error(request, exc):
        log.warning("decision read failed: %s (cause: %r)", exc, exc.__cause__)
        if isinstance(exc.__cause__, httpx.TimeoutException):
            return failure(request, 504, TIMEOUT, "vLLM did not answer in time")
        return failure(request, 502, BACKEND_UNAVAILABLE, str(exc))

    return failure


def build_remote_app(
    engine: RemoteVllmEngine,
    chat_model: str | None = None,
    error_format: str = "default",
    generation_limits: GenerationLimits | None = None,
):
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from starlette.background import BackgroundTask

    app = build_app(engine, error_format)
    app.title = "SystemOne — choice, noul, score, labels"
    # Keep our typed /v1/systemone; forward the ordinary OpenAI API unchanged.
    app.router.routes[:] = [
        r
        for r in app.router.routes
        if getattr(r, "path", None) not in {"/v1/models", "/health", "/ping"}
    ]
    # Loopback backend: ignore HTTP_PROXY. Non-streaming chat sends nothing until the generation ends,
    # so the read timeout is long; connecting fails fast.
    proxy = httpx.AsyncClient(
        timeout=httpx.Timeout(180, connect=10, read=3600), trust_env=False
    )
    failure = install_backend_errors(app, error_format)

    @app.get("/health")
    @app.get("/ping")
    async def health():
        try:
            response = await proxy.get(engine.backend_url + "/health")
            response.raise_for_status()
        except httpx.HTTPError:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "backend_unavailable",
                    "error": "vLLM backend is not answering",
                    "code": BACKEND_UNAVAILABLE,
                },
            )
        return {
            "status": "ok",
            "calibrated": bool(engine.temperatures),
            "calibration": Path(engine.calibration_source).name
            if engine.calibration_source
            else None,
        }

    @app.get("/v1/models")
    async def models(request: Request):
        try:
            response = await proxy.get(engine.backend_url + "/v1/models")
        except httpx.HTTPError:
            return failure(request, 502, BACKEND_UNAVAILABLE, "vLLM unavailable")
        content = response.content
        if response.is_success:
            # Decision names answer only /v1/systemone; chat and completions reject them.
            try:
                listing = response.json()
                listing["data"] = [
                    m for m in listing["data"] if m.get("id") not in engine.aliases
                ]
                content = json.dumps(listing).encode()
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        return Response(
            content, status_code=response.status_code, media_type="application/json"
        )

    limits = generation_limits or GenerationLimits(context_tokens=getattr(engine, "max_model_len", CONTEXT_TOKENS))

    async def forward(request: Request):
        try:
            body = public_request_body(
                await request.body(), engine.aliases, chat_model, limits, request.url.path
            )
        except RequestError as exc:
            return failure(request, 400, INVALID_REQUEST, exc.msg)
        try:
            response = await proxy.send(
                proxy.build_request(
                    "POST",
                    engine.backend_url + request.url.path,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept-Encoding": "identity",
                        **({"X-Request-Id": rid} if (rid := backend_request_id()) else {}),
                    },
                ),
                stream=True,
            )
        except httpx.TimeoutException:
            return failure(request, 504, TIMEOUT, "vLLM did not answer in time")
        except httpx.HTTPError:
            return failure(request, 502, BACKEND_UNAVAILABLE, "vLLM unavailable")
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            background=BackgroundTask(response.aclose),
        )

    app.add_api_route("/v1/chat/completions", forward, methods=["POST"])
    app.add_api_route("/v1/completions", forward, methods=["POST"])

    async def shutdown():
        await proxy.aclose()
        engine.close()

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            try:
                yield
            finally:
                await shutdown()

    app.router.lifespan_context = lifespan
    return app


def add_calibration_arguments(parser):
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration.json (default: next to --metadata, if present)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="return raw temperature-1 probabilities",
    )


def _positive_int_or_none(text: str) -> int | None:
    """argparse type: a whole number of at least 1, or "none" to lift the limit."""
    if text.strip().lower() == "none":
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1 (or none)")
    return value


def add_generation_arguments(parser):
    parser.add_argument(
        "--max-generation-n",
        type=_positive_int_or_none,
        default=DEFAULT_MAX_GENERATION_N,
        help="largest n (or best_of, the beam width) one chat/completions request may ask for; "
        f"'none' lifts it (default: {DEFAULT_MAX_GENERATION_N})",
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=_positive_int_or_none,
        default=DEFAULT_MAX_GENERATION_TOKENS,
        help="output tokens one chat/completions request may ask for in all (prompts x n x max_tokens, an unset "
        f"max_tokens counting as the full context); 'none' lifts it (default: {DEFAULT_MAX_GENERATION_TOKENS})",
    )


def generation_limits(args, context_tokens: int = CONTEXT_TOKENS) -> GenerationLimits:
    return GenerationLimits(args.max_generation_n, args.max_generation_tokens, context_tokens)


def main():
    import uvicorn
    from bielik_one.api.reasoning import EndpointReasoner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8013")
    parser.add_argument("--backend-model", default="decision")
    parser.add_argument("--model-name", default="decision")
    parser.add_argument("--base-model", default="qwen-base")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent backend reads (default: 8)"
    )
    add_calibration_arguments(parser)
    add_guard_arguments(parser)
    add_generation_arguments(parser)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    table, source = resolve_calibration(
        args.metadata,
        args.calibration,
        args.no_calibration,
        metadata.get("source_repo"),
    )
    print(f"calibration: {source or 'none (temperature 1)'}", flush=True)
    engine = RemoteVllmEngine(
        metadata,
        args.backend_url,
        model_name=args.model_name,
        backend_model=args.backend_model,
        workers=args.workers,
        calibration=table,
        calibration_source=source,
    )
    engine.reasoners["original"] = EndpointReasoner(
        args.backend_url + "/v1", args.base_model, tok=getattr(engine, "tok", None)
    )
    uvicorn.run(
        guarded(
            build_remote_app(
                engine, chat_model=args.base_model, error_format=args.error_format,
                generation_limits=generation_limits(args, getattr(engine, "max_model_len", CONTEXT_TOKENS)),
            ),
            args.max_body_mb,
            args.error_format,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
