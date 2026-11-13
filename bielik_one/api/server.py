"""HTTP server for the three softmax letter readouts.

One request is one or more prompts that differ only in option order. vLLM
keeps the shared prefix (system, state, question) in the prefix cache. A
logits mask drops every token that is not a label, so the returned logprobs
are the training letter softmax. Permutations are averaged at temperature 1;
a calibration temperature, if the caller sends one, is applied to that mean.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bielik_one.api.labels_park import sigmoid_from_logprobs
from bielik_one.prompting.intervals import at_temperature
from bielik_one.prompting.letter import answer_token_ids
from bielik_one.prompting.orders import _alt_orders, _reordered
from bielik_one.prompting.schema import (
    CHOICE_MAX,
    SCORE_MAX,
    SCORE_MIN,
    Option,
    Question,
    Row,
)

KINDS = ("choice", "noul", "score")
MAX_PERMUTATIONS = 16
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 400_000  # well above a 32768-token context; longer prompts are rejected by the backend anyway
MAX_QUESTIONS_CHARS = 400_000  # all question names, instructions, option names and texts together
MAX_PROMPTS = MAX_QUESTIONS * MAX_PERMUTATIONS  # decision reads per request, after per-option expansion
# Per-option prompts repeat the question and the option listing once per option; this caps the text they add per
# request (checked before any of it is built).
MAX_PER_OPTION_CHARS = 2_000_000
LABEL_MODES = ("head", "per_option")
# Per-option labels: one yes/no question per option, each seeing the question, every option and one candidate. The
# wording is the per-option protocol measured in the technical report (labels head versus per-option yes/no).
PER_OPTION_DECISION = "Does this candidate belong to the correct answer set?"


class RequestError(ValueError):
    """A body the System One schema rejects. `loc` is the field path after `body`."""

    def __init__(self, loc: list[str | int], msg: str):
        super().__init__(msg)
        self.loc = ["body", *loc]
        self.msg = msg


class QuestionIn(BaseModel):
    """One TypeSafe question. `criteria` is a map for choice, a list for score, and `{true, false}` for noul."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["choice", "noul", "score", "labels"]
    instructions: Any = None
    criteria: Any = None
    # labels only: "head" (default) reads every option in one pass from the labels head; "per_option" asks one noul
    # question per option and returns the same labels answer. Not part of the TypeSafe schema (labels is our own type).
    mode: Literal["head", "per_option"] = "head"


def _reject_lone_surrogates(value: Any, path: Sequence[str | int] = ()) -> None:
    """JSON can spell a lone UTF-16 surrogate ("\\udead"); it is not Unicode text, so the tokenizer and the UTF-8
    response would fail on it. Raises ValueError naming the first string (or key) that holds one."""
    stack = [(value, list(path))]
    while stack:  # iterative: a deeply nested value must not exhaust the recursion limit
        item, where = stack.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                place = ".".join(str(p) for p in where) or "value"
                place = place.encode("utf-8", "backslashreplace").decode("utf-8")  # a key may hold one too
                raise ValueError(f"{place} contains a lone UTF-16 surrogate, which is not valid Unicode text") from None
        elif isinstance(item, dict):
            for key, child in item.items():
                stack.append((key, where))
                stack.append((child, [*where, key]))
        elif isinstance(item, (list, tuple)):
            stack.extend((child, [*where, i]) for i, child in enumerate(item))


class SystemOneIn(BaseModel):
    """`POST /v1/systemone`. `permutations`, `seed`, and `temperature` are optional and absent from the TypeSafe schema."""

    model_config = ConfigDict(extra="ignore")

    state: Any
    model: str = Field(min_length=1)
    questions: dict[str, QuestionIn] = Field(min_length=1, max_length=MAX_QUESTIONS)
    permutations: int = Field(default=1, ge=1, le=MAX_PERMUTATIONS)
    seed: int | None = None
    temperature: float | dict[str, float] | None = None
    questions_before_state: bool = False
    read_twice: bool = True
    read_twice_max_input_tokens: int | None = Field(default=4096, ge=0, strict=True)
    reasoning: Literal["off", "on", "auto"] = "off"
    reasoning_model: Literal["original"] = "original"
    reasoning_threshold: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    reasoning_max_tokens: int = Field(default=256, ge=1, le=8192)

    @field_validator("state", "model")
    @classmethod
    def unicode_text(cls, value, info):
        _reject_lone_surrogates(value, [info.field_name])
        return value

    @field_validator("questions")
    @classmethod
    def bounded_questions(cls, value):
        for name, question in value.items():
            _reject_lone_surrogates(name, [name])
            _reject_lone_surrogates(question.instructions, [name, "instructions"])
            _reject_lone_surrogates(question.criteria, [name, "criteria"])
        size = sum(
            len(name) + len(json.dumps([q.instructions, q.criteria], ensure_ascii=False))
            for name, q in value.items()
        )
        if size > MAX_QUESTIONS_CHARS:
            raise ValueError(f"questions are longer than {MAX_QUESTIONS_CHARS} characters in total")
        return value

    @field_validator("temperature")
    @classmethod
    def finite_positive_temperature(cls, value):
        values = value.values() if isinstance(value, dict) else [value]
        for item in values:
            if item is not None and not (math.isfinite(item) and item > 0):
                raise ValueError("temperature must be a finite number greater than 0")
        return value

    @field_validator("reasoning", mode="before")
    @classmethod
    def reasoning_alias(cls, value):
        return (
            {"never": "off", "always": "on", "threshold": "auto"}.get(value, value)
            if isinstance(value, str)
            else value
        )


def _logprob_value(entry: Any) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, Mapping):
        return float(entry["logprob"])
    return float(entry)


def probs_from_logprobs(table: Mapping, letter_ids: Sequence[int]) -> list[float]:
    """Letter softmax, read by token id. A missing id means the mask or the logprob cap failed."""
    values = []
    for token_id in letter_ids:
        key = int(token_id)
        entry = table.get(key)
        if entry is None:
            entry = table.get(str(key))
        if entry is None:
            raise RuntimeError(
                f"label token {key} is missing from the logprobs; "
                f"the server must be started with max_logprobs>={len(letter_ids)} and a letter mask"
            )
        values.append(math.exp(_logprob_value(entry)))
    total = sum(values)
    if not all(math.isfinite(value) for value in values) or not math.isclose(
        total, 1.0, abs_tol=1e-3
    ):
        raise RuntimeError(
            f"answer probabilities sum to {total}; vLLM must use logprobs_mode=processed_logprobs"
        )
    return [value / total for value in values]


def option_orders(n: int, k: int, seed: int) -> list[list[int]]:
    """`k == 1` is the request order, `k == 2` adds the exact reversal, and `k > 2` adds random orders.

    The result can be shorter than `k` when there are not that many distinct permutations.
    """
    identity = list(range(n))
    if k <= 1 or n < 2:
        return [identity]
    if k == 2:
        return [identity, list(reversed(identity))]
    extra = _alt_orders(n, k - 1, random.Random(seed))
    return [identity, *[list(perm) for perm in extra]]


def align(probs: Sequence[float], perm: Sequence[int]) -> list[float]:
    """Map a distribution scored on `_reordered(row, perm)` back onto the original options."""
    out = [0.0] * len(probs)
    for position, origin in enumerate(perm):
        out[origin] = float(probs[position])
    return out


def average(rows: Sequence[Sequence[float]]) -> list[float]:
    width = len(rows[0])
    count = len(rows)
    return [sum(row[i] for row in rows) / count for i in range(width)]


def apply_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    """`softmax(log p / T)` via `at_temperature`. One temperature does not move the argmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    tensor = at_temperature(
        torch.tensor(list(probs), dtype=torch.float32), float(temperature)
    )
    return [float(value) for value in tensor]


def resolve_temperature(
    requested: float | Mapping[str, float] | None,
    kind: str,
    saved: Mapping | None = None,
    permutations: int = 1,
) -> float:
    """A request temperature wins. Otherwise use the export table for one order, or for two or more."""
    if requested is None:
        return _saved_temperature(saved, kind, permutations)
    if isinstance(requested, Mapping):
        if kind not in requested:
            return 1.0
        value = float(requested[kind])
    else:
        value = float(requested)
    if value <= 0:
        raise ValueError("temperature must be positive")
    return value


def _saved_temperature(saved: Mapping | None, kind: str, permutations: int) -> float:
    if not isinstance(saved, Mapping):
        return 1.0
    # A table fitted for two or more orders wins when present; otherwise the one-order temperatures apply, which
    # measured better than raw probabilities with 2 and 4 orders (native validation, Bielik-One-4B-r1).
    table = saved.get("2") if permutations > 1 and isinstance(saved.get("2"), Mapping) else saved.get("1")
    if not isinstance(table, Mapping) or kind not in table:
        return 1.0
    value = float(table[kind])
    if value <= 0:
        raise ValueError("temperature must be positive")
    return value


def apply_sigmoid_temperature(
    probs: Sequence[float], temperature: float
) -> list[float]:
    """Divide each label log-odds by `temperature`. A probability of 0.5 stays 0.5."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if abs(temperature - 1.0) < 1e-8:
        return [float(value) for value in probs]
    scaled = []
    for prob in probs:
        clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
        logit = math.log(clipped) - math.log(1.0 - clipped)
        value = logit / temperature
        if value >= 0:
            scaled.append(1.0 / (1.0 + math.exp(-value)))
        else:
            exponential = math.exp(value)
            scaled.append(exponential / (1.0 + exponential))
    return scaled


def _rendered(value: Any) -> str:
    """Text the letter prompt can hold. Objects and arrays become JSON; null stays empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError("expected a string, object, or array")


def state_text(state: Any) -> str:
    """A string state is kept as sent. An object or array is JSON, the same dump the scorer uses."""
    if isinstance(state, str):
        if state and not state.strip():
            raise RequestError(["state"], "state is whitespace only")
        text = state
    elif isinstance(state, (dict, list)):
        text = json.dumps(state, ensure_ascii=False)
    else:
        raise RequestError(["state"], "state must be a string, object, or array")
    if len(text) > MAX_STATE_CHARS:
        raise RequestError(["state"], f"state is longer than {MAX_STATE_CHARS} characters")
    return text


def _instructions(name: str, value: Any) -> str:
    if value is None:
        raise RequestError(
            ["questions", name, "instructions"], "instructions is required"
        )
    try:
        text = _rendered(value)
    except ValueError:
        raise RequestError(
            ["questions", name, "instructions"],
            "instructions must be a string, object, or array",
        ) from None
    if isinstance(value, str) and not text.strip():
        raise RequestError(["questions", name, "instructions"], "instructions is blank")
    return text


def question_to_row(
    state: str, name: str, question: QuestionIn
) -> tuple[Row, list[Any] | None]:
    """A System One question as a training row. The second value is the score legend, in request order."""
    kind = question.type
    if question.mode != "head" and kind != "labels":
        raise RequestError(["questions", name, "mode"], "mode applies to labels questions only")
    instructions = _instructions(name, question.instructions)
    loc = ["questions", name, "criteria"]
    legend: list[Any] | None = None
    if kind == "noul":
        criteria = question.criteria or {}
        if not isinstance(criteria, dict):
            raise RequestError(
                loc, "noul criteria must be an object with true and false"
            )
        try:
            options = [
                Option(key="no", text=_rendered(criteria.get("false"))),
                Option(key="yes", text=_rendered(criteria.get("true"))),
            ]
        except ValueError:
            raise RequestError(
                loc,
                "noul criteria true and false must be a string, object, array, or null",
            ) from None
        stored: dict[str, str] | list[str] = {
            option.key: option.text for option in options
        }
    elif kind in ("choice", "labels"):
        criteria = question.criteria
        if not isinstance(criteria, dict):
            raise RequestError(
                loc, f"{kind} criteria must map option names to descriptions"
            )
        if kind == "choice" and len(criteria) < 2:
            raise RequestError(loc, "a choice needs at least two options")
        if kind == "labels" and len(criteria) < 2:
            raise RequestError(loc, "a labels question needs at least two options")
        if len(criteria) > CHOICE_MAX:
            raise RequestError(
                loc, f"{len(criteria)} options; the limit is {CHOICE_MAX}"
            )
        options = []
        for key, text in criteria.items():
            if not isinstance(key, str) or not key:
                raise RequestError(loc, "choice option names must be non-empty strings")
            try:
                text_out = _rendered(text)
            except ValueError:
                raise RequestError(
                    loc, "a choice description must be a string, object, array, or null"
                ) from None
            options.append(Option(key=key, text=text_out))
        stored = {option.key: option.text for option in options}
    else:
        criteria = question.criteria
        if not isinstance(criteria, list):
            raise RequestError(loc, "a score takes 2 to 10 levels")
        if not SCORE_MIN <= len(criteria) <= SCORE_MAX:
            raise RequestError(loc, "a score takes 2 to 10 levels")
        legend = []
        texts = []
        for item in criteria:
            if not isinstance(item, (str, dict, list)):
                raise RequestError(
                    loc, "a score level must be a string, object, or array"
                )
            legend.append(item)
            texts.append(_rendered(item))
        options = [Option(key=str(i), text=text) for i, text in enumerate(texts)]
        stored = texts
    n = len(options)
    try:
        return _row(name, state, kind, instructions, stored, options, n), legend
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        raise RequestError(
            ["questions", name], str(first.get("msg", "invalid question"))
        ) from None


def per_option_questions(row: Row) -> list[tuple[str, QuestionIn]]:
    """A labels question as one noul question per option, in option order: (option key, question)."""
    listing = "\n".join(f"- {option.key}: {option.text}" for option in row.options)
    return [
        (
            option.key,
            QuestionIn(
                type="noul",
                instructions=f"{row.question.instructions}\n\nAll options:\n{listing}\n\n"
                f"Candidate: {option.key}: {option.text}\n{PER_OPTION_DECISION}",
                criteria={"true": "yes", "false": "no"},
            ),
        )
        for option in row.options
    ]


def _row(name, state, kind, instructions, stored, options, n) -> Row:
    return Row(
        id=name,
        split="test",
        lang="und",
        source="serve",
        license="unspecified",
        group_id=name,
        state=state,
        question=Question(type=kind, instructions=instructions, criteria=stored),
        options=options,
        target_probs=[1.0] + [0.0] * (n - 1),
    )


def choice_confidence(probs: Sequence[float]) -> float:
    """How far the top probability is above the uniform 1/K, on [0, 1]: the reference System One formula (Hugging Face
    Transformers' decision pipeline, llama.cpp). 0 for a uniform distribution, 1 when one option has it all."""
    n = len(probs)
    if n == 1:
        return 1.0
    return max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def score_confidence(probs: Sequence[float]) -> float:
    """1 - the expected distance (in levels) from the modal level, relative to that of a uniform distribution from
    the middle of the scale: the reference System One formula. A near miss costs less than a far one."""
    n = len(probs)
    if n == 1:
        return 1.0
    best = max(range(n), key=lambda i: probs[i])
    distance = sum(p * abs(level - best) for level, p in enumerate(probs))
    uniform = sum(abs(level - (n - 1) / 2) for level in range(n)) / n
    return max(0.0, 1 - distance / uniform)


def systemone_answer(
    kind: str,
    keys: Sequence[str],
    probs: Sequence[float],
    legend: Sequence[Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """One answer in the System One shape. Noul is the probability of yes; every answer has its probabilities.

    Confidence follows the reference implementation, so the same request gets the same confidence on any System One
    server: choice_confidence for a choice (and for a noul, as a choice between its two sides), score_confidence for a
    score. It summarizes how concentrated the distribution is; it is not a probability of being right, which is the
    chosen answer's entry in `probabilities`. Labels (not a reference type) report the probability that the selected
    set is exactly right, with the options taken as independent.
    """
    if kind == "noul":
        yes = float(probs[keys.index("yes")])
        no = float(probs[keys.index("no")])
        return {
            "type": "noul",
            "noul": yes,
            "probabilities": {"true": yes, "false": no},
            "confidence": choice_confidence([yes, no]),
        }
    if kind == "choice":
        paired = {key: float(prob) for key, prob in zip(keys, probs, strict=True)}
        chosen = max(paired, key=paired.__getitem__)
        return {
            "type": "choice",
            "choice": chosen,
            "probabilities": paired,
            "confidence": choice_confidence(list(paired.values())),
        }
    if kind == "labels":
        paired = {key: float(prob) for key, prob in zip(keys, probs, strict=True)}
        selected = [key for key, prob in paired.items() if prob >= threshold]
        # The probability that the whole selected set is exactly right, with the options independent: every
        # selected option applies and every other one does not.
        confidence = math.prod(
            prob if prob >= threshold else 1 - prob for prob in paired.values()
        )
        return {
            "type": "labels",
            "probabilities": paired,
            "selected": selected,
            "confidence": confidence,
        }
    levels = {str(i): float(prob) for i, prob in enumerate(probs)}
    shown = list(legend) if legend is not None else list(keys)
    return {
        "type": "score",
        "score": sum(i * float(prob) for i, prob in enumerate(probs)),
        "legend": {str(i): shown[i] for i in range(len(probs))},
        "probabilities": levels,
        "confidence": score_confidence([float(p) for p in probs]),
    }


def score_orders(
    logprob_tables: Sequence[Mapping],
    letter_ids: Sequence,
    orders: Sequence[Sequence[int]],
    temperature: float,
) -> list[float]:
    """Average the permutations at temperature 1, then apply `temperature` to the mean.

    `letter_ids` is one id list when every order uses the same positional tokens,
    or one list per order when the token travels with the option.
    """
    shared = bool(letter_ids) and isinstance(letter_ids[0], int)
    aligned = []
    for index, (table, perm) in enumerate(zip(logprob_tables, orders, strict=True)):
        ids = letter_ids if shared else letter_ids[index]
        aligned.append(align(probs_from_logprobs(table, ids), perm))
    return apply_temperature(average(aligned), temperature)


class LetterEngine:
    """Decision logic shared by the vLLM frontends. Subclasses set tok, model_name, capacity, parking,
    labels_threshold, semantic, temperatures, max_model_len and reasoners, and implement distributions()."""

    def _evaluate_once(
        self,
        body: SystemOneIn,
        *,
        questions: Sequence[Row] = (),
        analyses: Mapping[str, str] | None = None,
        raw_confidence: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Every question on this state, in one batch. Permutations of one question stay next to each other.

        `raw_confidence`, when given, is filled with each answer's confidence at temperature 1 from the same
        forward pass, so reasoning gates do not depend on the calibration temperature.
        """
        state = state_text(body.state)
        seed = 0 if body.seed is None else body.seed
        prepared = []
        prompts: list[Row] = []
        reads = 0
        per_option_chars = 0
        for name, question in body.questions.items():
            row, legend = question_to_row(state, name, question)
            per_option = question.type == "labels" and question.mode == "per_option"
            # Count reads (and per-option text) before building any prompt, so an oversized request costs nothing.
            n = len(row.options)
            reads += (n if per_option else 1) * len(
                option_orders(2 if per_option else n, body.permutations, seed)
            )
            if reads > MAX_PROMPTS:
                raise RequestError(
                    ["questions"],
                    f"more than {MAX_PROMPTS} decision reads (questions x option orders, per-option labels "
                    f"count once per option); the limit is {MAX_PROMPTS}",
                )
            if per_option:
                listing = sum(len(option.key) + len(option.text) + 4 for option in row.options)
                per_option_chars += n * (len(row.question.instructions) + listing + 120)
                if per_option_chars > MAX_PER_OPTION_CHARS:
                    raise RequestError(
                        ["questions", name],
                        "per-option labels repeat the question and all options once per option; this request "
                        f"would add more than {MAX_PER_OPTION_CHARS} characters. Use \"mode\": \"head\" or fewer, "
                        "shorter options",
                    )
            if per_option:
                # One noul read per option (each in its own option orders); answered as one labels question.
                parts = []
                for key, sub in per_option_questions(row):
                    sub_row, _ = question_to_row(state, name, sub)
                    orders = option_orders(2, body.permutations, seed)
                    parts.append((key, sub_row, orders))
                    prompts.extend(_reordered(sub_row, perm) for perm in orders)
                prepared.append((name, row, legend, parts))
                continue
            if len(row.options) > self.capacity:
                raise RequestError(
                    ["questions", name, "criteria"],
                    f"checkpoint supports at most {self.capacity} options",
                )
            orders = option_orders(len(row.options), body.permutations, seed)
            prepared.append((name, row, legend, orders))
            prompts.extend(_reordered(row, perm) for perm in orders)
        assert len(prompts) == reads
        if body.questions_before_state or body.read_twice or analyses:
            tables, input_tokens = self.distributions(
                prompts,
                questions=questions or [item[1] for item in prepared],
                questions_before_state=body.questions_before_state,
                read_twice=body.read_twice,
                read_twice_max_input_tokens=body.read_twice_max_input_tokens,
                analyses=analyses,
            )
        else:
            tables, input_tokens = self.distributions(prompts)
        answers = {}
        cursor = 0
        for name, row, legend, orders in prepared:
            # The temperatures were fitted on the standard prompt with one option order and apply to every request
            # mode: refitting them for read_twice (and FP8) lowered test NLL by at most 0.012, and with 2 or 4 option
            # orders or questions_before_state they beat raw probabilities on every categorical type (native
            # validation, Bielik-One-4B-r1). Only the reasoning pass, which was not measured, stays at temperature 1.
            saved = {} if analyses else self.temperatures
            if body.questions[name].mode == "per_option":
                # Each option is a noul decision, so it takes the noul temperature and the 0.5 noul threshold.
                temperature = resolve_temperature(body.temperature, "noul", saved, body.permutations)
                keys, probs, raw = [], [], []
                for key, sub_row, sub_orders in orders:
                    chunk = tables[cursor : cursor + len(sub_orders)]
                    cursor += len(sub_orders)
                    token_ids = [
                        answer_token_ids(self.tok, _reordered(sub_row, perm), self.semantic)
                        for perm in sub_orders
                    ]
                    yes = [option.key for option in sub_row.options].index("yes")
                    keys.append(key)
                    probs.append(score_orders(chunk, token_ids, sub_orders, temperature)[yes])
                    if raw_confidence is not None:
                        raw.append(score_orders(chunk, token_ids, sub_orders, 1.0)[yes])
                answers[name] = {**systemone_answer("labels", keys, probs, legend, 0.5), "mode": "per_option"}
                if raw_confidence is not None:
                    raw_confidence[name] = answer_confidence(
                        systemone_answer("labels", keys, raw, legend, 0.5)
                    )
                continue
            chunk = tables[cursor : cursor + len(orders)]
            cursor += len(orders)
            temperature = resolve_temperature(
                body.temperature, row.question.type, saved, body.permutations
            )
            keys = [option.key for option in row.options]
            if row.question.type == "labels":
                n = len(row.options)
                option_ids = [
                    int(token_id) for token_id in self.parking["option_ids"][:n]
                ]
                anchor = int(self.parking["anchor_id"])
                aligned = [
                    align(sigmoid_from_logprobs(table, option_ids, anchor), perm)
                    for table, perm in zip(chunk, orders, strict=True)
                ]
                probs = apply_sigmoid_temperature(average(aligned), temperature)
                answers[name] = systemone_answer(
                    row.question.type, keys, probs, legend, self.labels_threshold
                )
                if raw_confidence is not None:
                    raw_confidence[name] = answer_confidence(
                        systemone_answer(
                            row.question.type,
                            keys,
                            average(aligned),
                            legend,
                            self.labels_threshold,
                        )
                    )
            else:
                token_ids = [
                    answer_token_ids(self.tok, _reordered(row, perm), self.semantic)
                    for perm in orders
                ]
                probs = score_orders(chunk, token_ids, orders, temperature)
                answers[name] = systemone_answer(row.question.type, keys, probs, legend)
                if raw_confidence is not None:
                    raw_confidence[name] = answer_confidence(
                        systemone_answer(
                            row.question.type,
                            keys,
                            score_orders(chunk, token_ids, orders, 1.0),
                            legend,
                        )
                    )
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": len(tables)},
        }

    # Concurrent analyses per request; the backend batches them (default profile: 8 sequences).
    reasoning_parallel = 8

    def _reason(self, reasoner, rows, max_tokens: int) -> dict[str, dict]:
        """Analyses for `rows`, generated concurrently; keyed and ordered like `rows`. The first failure is raised
        and analyses not yet started are cancelled."""
        if len(rows) <= 1:
            return {row.id: reasoner(row, max_tokens) for row in rows}
        pool = ThreadPoolExecutor(max_workers=min(len(rows), self.reasoning_parallel))
        try:
            futures = [pool.submit(reasoner, row, max_tokens) for row in rows]
            return {row.id: future.result() for row, future in zip(rows, futures)}
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # Reasoning is disabled in 1.0: the current path (a separate analysis pasted into the decision prompt) lowered
    # validation accuracy and NLL in a quick check. It will return as a thought in <think> on the decision model.
    reasoning_enabled = False

    def evaluate(self, body: SystemOneIn) -> dict[str, Any]:
        if body.reasoning == "off":
            return self._evaluate_once(body)
        if not self.reasoning_enabled:
            raise RequestError(
                ["reasoning"], "reasoning is not available in this release; use \"off\""
            )
        reasoner = getattr(self, "reasoners", {}).get(body.reasoning_model)
        if reasoner is None:
            raise RequestError(
                ["reasoning_model"],
                f"no {body.reasoning_model} reasoning backend configured",
            )
        state = state_text(body.state)
        rows = [
            question_to_row(state, name, question)[0]
            for name, question in body.questions.items()
        ]
        # Validate the complete decision batch before starting costly generation.
        for row in rows:
            if body.questions[row.id].mode == "per_option":
                continue  # read as noul questions: no labels head and no option capacity needed
            if len(row.options) > self.capacity:
                raise RequestError(
                    ["questions", row.id, "criteria"],
                    f"checkpoint supports at most {self.capacity} options",
                )
            if row.question.type == "labels" and not self.parking:
                raise RequestError(
                    ["questions", row.id], "this checkpoint has no labels head"
                )
        # The threshold applies to raw (temperature 1) confidence, as before calibration was introduced;
        # calibrated temperatures would otherwise send far more questions to the reasoner.
        gate: dict[str, float] = {}
        first = (
            self._evaluate_once(body, raw_confidence=gate)
            if body.reasoning == "auto"
            else None
        )
        selected = [
            row
            for row in rows
            if first is None or gate[row.id] < body.reasoning_threshold
        ]
        records = self._reason(reasoner, selected, body.reasoning_max_tokens)
        if selected:
            subset = body.model_copy(
                update={
                    "questions": {row.id: body.questions[row.id] for row in selected}
                }
            )
            result = self._evaluate_once(
                subset,
                questions=rows,
                analyses={key: value["text"] for key, value in records.items()},
            )
        else:
            result = {
                "model": self.model_name,
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        if first is not None:
            result["answers"] = {
                key: result["answers"].get(key, value)
                for key, value in first["answers"].items()
            }
            for key in ("input_tokens", "output_tokens"):
                result["usage"][key] += first["usage"][key]
        for record in records.values():
            for key in ("input_tokens", "output_tokens"):
                result["usage"][key] += record[key]
        result["inference"] = {
            "reasoning": body.reasoning,
            "reasoning_model": body.reasoning_model,
            "reasoning_threshold": body.reasoning_threshold,
            "analyses": records,
        }
        return result


def answer_confidence(answer: Mapping) -> float:
    return float(answer["confidence"])


def build_app(engine: "LetterEngine", error_format: str = "default"):
    """Typed decision endpoint for a ready engine. The vLLM frontend adds health, models and chat routes."""
    from fastapi import FastAPI, HTTPException

    from bielik_one.api.errors import install

    app = FastAPI(title="Bielik One decisions")
    install(app, error_format)

    @app.post("/v1/systemone")
    def systemone(body: SystemOneIn) -> dict[str, Any]:
        try:
            return engine.evaluate(body)
        except RequestError as exc:
            raise HTTPException(
                status_code=422,
                detail=[{"loc": exc.loc, "msg": exc.msg, "type": "value_error"}],
            ) from exc

    return app
