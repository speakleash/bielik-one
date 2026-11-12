"""One question per JSONL row. Packing several questions over one state is a collator choice.

`gold_index` is not stored. It is `argmax(target_probs)`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Split = Literal["train", "validation", "calibration", "test", "ood", "test_set_30k"]
QuestionType = Literal["choice", "noul", "score", "labels"]

NOUL_KEYS = ("no", "yes")
SCORE_MIN, SCORE_MAX = 2, 10
CHOICE_MAX = 255
PROB_SUM_TOL = 1e-3


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    text: str


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("instructions")
    @classmethod
    def _instructions_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions is blank")
        return value


class Row(BaseModel):
    """Canonical training row. `criteria` is the TypeSafe question body, so a row renders into `/v1/systemone`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: Split
    lang: str = Field(min_length=2, max_length=16)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    state: str
    question: Question
    options: list[Option] = Field(min_length=2)
    target_probs: list[float]
    teacher: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def _state_not_blank(cls, value: str) -> str:
        if value and not value.strip():
            raise ValueError("state is whitespace only")
        return value

    @model_validator(mode="after")
    def _aligned(self) -> Row:
        keys = [o.key for o in self.options]
        if len(keys) != len(set(keys)):
            raise ValueError("option keys must be unique")
        if len(self.target_probs) != len(self.options):
            raise ValueError("target_probs must align with options")
        if any(p < 0 or p != p or p == float("inf") for p in self.target_probs):
            raise ValueError("target_probs must be finite and non-negative")
        kind = self.question.type
        if kind != "labels":
            total = sum(self.target_probs)
            if abs(total - 1.0) > PROB_SUM_TOL:
                raise ValueError(f"target_probs sum to {total}, expected 1")
        if kind == "noul":
            if tuple(keys) != NOUL_KEYS:
                raise ValueError("noul options must be ordered no, yes")
        elif kind in ("choice", "labels"):
            if len(self.options) > CHOICE_MAX:
                raise ValueError(
                    f"{kind} has {len(self.options)} options; the limit is {CHOICE_MAX}"
                )
            if kind == "labels" and any(p > 1 for p in self.target_probs):
                raise ValueError("labels targets must be in [0, 1]")
            if not isinstance(self.question.criteria, dict):
                raise ValueError(f"{kind} criteria must be {{key: description}}")
            if list(self.question.criteria) != keys:
                raise ValueError(f"{kind} criteria keys must match options in order")
            if list(self.question.criteria.values()) != [o.text for o in self.options]:
                raise ValueError(f"{kind} criteria values must match option texts")
        elif kind == "score":
            if not SCORE_MIN <= len(self.options) <= SCORE_MAX:
                raise ValueError(f"score needs {SCORE_MIN} to {SCORE_MAX} levels")
            if not isinstance(self.question.criteria, list):
                raise ValueError(
                    "score criteria must be a list of level strings, low to high"
                )
            if self.question.criteria != [o.text for o in self.options]:
                raise ValueError("score criteria must match option texts in order")
            if keys != [str(i) for i in range(len(self.options))]:
                raise ValueError(
                    "score option keys must be the level indices as strings"
                )
        else:
            raise ValueError(f"unknown question type: {kind}")
        return self

    @property
    def gold_index(self) -> int:
        if self.question.type == "labels":
            raise ValueError("a labels row has no single gold option")
        return max(range(len(self.target_probs)), key=self.target_probs.__getitem__)

    @property
    def gold_key(self) -> str:
        return self.options[self.gold_index].key


def span_text(option: Option) -> str:
    """Text the encoder sees. An empty description falls back to the key."""
    return option.text if option.text.strip() else option.key


def content_hash(row: Row) -> str:
    """Normalized `(state, question, sorted option texts)`. Dedup compares this across corpora.

    JSON states are hashed in canonical form, so a pretty-printed Open-Jev state matches the
    compact JSON stored for the same scenario in jev-distill.
    """
    payload = json.dumps(
        {
            "state": _norm_state(row.state),
            "question": _norm(row.question.instructions),
            "options": sorted(_norm(span_text(o)) for o in row.options),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def systemone_question(row: Row) -> dict[str, Any]:
    """The question object a `/v1/systemone` request sends for this row."""
    body: dict[str, Any] = {
        "type": row.question.type,
        "instructions": row.question.instructions,
    }
    if row.question.type in ("choice", "labels"):
        body["criteria"] = {o.key: o.text for o in row.options}
    elif row.question.type == "score":
        body["criteria"] = [o.text for o in row.options]
    return body


def pack_record(rows: Sequence[Row]) -> dict[str, Any]:
    """Internal encoder record: one state, one branch per row. Rows must share `state`."""
    if not rows:
        raise ValueError("no rows")
    state = rows[0].state
    if any(r.state != state for r in rows):
        raise ValueError("rows in one pack must share a state")
    return {
        "state": state,
        "questions": [
            {
                "instr": r.question.instructions,
                "options": [span_text(o) for o in r.options],
                "target": list(r.target_probs),
                "type": r.question.type,
                "keys": [o.key for o in r.options],
            }
            for r in rows
        ],
    }


def load_jsonl(path: str | Path) -> list[Row]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[Row]:
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield Row.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


def dump_jsonl(path: str | Path, rows: Iterable[Row]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _norm_state(text: str) -> str:
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _norm(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(
                parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
    return _norm(text)
