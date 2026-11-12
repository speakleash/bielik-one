"""labels with "mode": "per_option": one noul read per option, answered in the labels shape."""

import math

import pytest

from bielik_one.api import server
from bielik_one.api.server import (
    MAX_PROMPTS,
    LetterEngine,
    QuestionIn,
    RequestError,
    SystemOneIn,
    per_option_questions,
    question_to_row,
)

YES = {"a": 0.9, "b": 0.2, "c": 0.6}


class PerOptionEngine(LetterEngine):
    """No labels head (parking None): per-option labels must not need one. Each prompt's letter table gives the
    candidate's yes probability from YES, by position, so option orders are exercised."""

    model_name = "decision"
    capacity = 255
    labels_threshold = 0.5
    semantic = True
    parking = None
    tok = None

    def __init__(self, temperatures=None):
        self.temperatures = temperatures or {}
        self.prompts = []

    def distributions(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        tables = []
        for row in prompts:
            text = row.question.instructions
            key = text.split("Candidate: ", 1)[1].split(":", 1)[0]
            p = {"yes": YES[key], "no": 1 - YES[key]}
            tables.append({20 + i: math.log(p[option.key]) for i, option in enumerate(row.options)})
        return tables, 7 * len(prompts)


@pytest.fixture(autouse=True)
def positional_tokens(monkeypatch):
    monkeypatch.setattr(server, "answer_token_ids", lambda tok, row, semantic: [20 + i for i in range(len(row.options))])


def body(mode="per_option", options=YES, **extra):
    return SystemOneIn(
        model="decision", state="An invoice of 4300 EUR is unpaid; the deadline passed yesterday.",
        questions={"actions": {"type": "labels", "instructions": "Which actions apply now?", "mode": mode,
                               "criteria": {k: f"Option {k}" for k in options}}},
        **extra,
    )


def test_per_option_answer_has_the_labels_shape():
    engine = PerOptionEngine()
    out = engine.evaluate(body())
    answer = out["answers"]["actions"]
    assert answer["type"] == "labels" and answer["mode"] == "per_option"
    assert answer["probabilities"] == pytest.approx(YES)
    assert answer["selected"] == ["a", "c"]
    assert answer["confidence"] == pytest.approx(0.9 * 0.8 * 0.6)
    assert len(engine.prompts) == 3 and out["usage"] == {"input_tokens": 21, "output_tokens": 3}
    assert all(row.question.type == "noul" for row in engine.prompts)


def test_permutations_average_both_orders():
    engine = PerOptionEngine()
    answer = engine.evaluate(body(permutations=2))["answers"]["actions"]
    assert len(engine.prompts) == 6
    assert answer["probabilities"] == pytest.approx(YES)


def test_per_option_uses_the_noul_temperature():
    engine = PerOptionEngine(temperatures={"1": {"noul": 2.0, "labels": 5.0}})
    answer = engine.evaluate(body())["answers"]["actions"]
    expected = {k: math.sqrt(p) / (math.sqrt(p) + math.sqrt(1 - p)) for k, p in YES.items()}
    assert answer["probabilities"] == pytest.approx(expected)
    assert answer["selected"] == ["a", "c"]  # a temperature does not move a 0.5 decision


def test_prompt_wording_lists_every_option_and_one_candidate():
    row, _ = question_to_row("state", "q", QuestionIn(type="labels", instructions="Pick", criteria={"x": "X", "y": "Y"}))
    parts = per_option_questions(row)
    assert [key for key, _ in parts] == ["x", "y"]
    text = parts[1][1].instructions
    assert text == "Pick\n\nAll options:\n- x: X\n- y: Y\n\nCandidate: y: Y\nDoes this candidate belong to the correct answer set?"
    assert parts[1][1].type == "noul" and parts[1][1].criteria == {"true": "yes", "false": "no"}


def test_head_mode_is_the_default_and_unchanged():
    assert QuestionIn(type="labels").mode == "head"
    assert body(mode="head").questions["actions"].mode == "head"


def test_mode_is_rejected_on_other_types():
    with pytest.raises(RequestError) as err:
        question_to_row("s", "q", QuestionIn(type="noul", instructions="ok?", mode="per_option"))
    assert err.value.loc == ["body", "questions", "q", "mode"]


def test_unknown_mode_fails_validation():
    with pytest.raises(Exception):
        body(mode="fast")


def test_expansion_counts_against_the_read_limit():
    # A noul has two option orders, so each option costs at most two reads; three questions of 200 options exceed it.
    criteria = {f"o{i}": f"Option {i}" for i in range(200)}
    questions = {f"q{j}": {"type": "labels", "instructions": "Pick", "mode": "per_option", "criteria": criteria} for j in range(3)}
    engine = PerOptionEngine()
    with pytest.raises(RequestError) as err:
        engine.evaluate(SystemOneIn(model="decision", state="s", questions=questions, permutations=16))
    assert err.value.loc == ["body", "questions"] and str(MAX_PROMPTS) in err.value.msg
    assert engine.prompts == []  # rejected before any backend read


def test_http_route_returns_the_answer():
    from fastapi.testclient import TestClient

    with TestClient(server.build_app(PerOptionEngine())) as client:
        response = client.post("/v1/systemone", json=body().model_dump())
    assert response.status_code == 200
    assert response.json()["answers"]["actions"]["selected"] == ["a", "c"]


class MixedEngine(PerOptionEngine):
    """Adds a labels head (parking) so head labels, per-option labels and choice share one batch."""

    parking = {"option_ids": [30, 31, 32], "anchor_id": 39}

    def distributions(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        tables = []
        for row in prompts:
            if row.question.type == "labels":  # head: logit 2 for every option -> sigmoid(2)
                tables.append({30: 2.0, 31: 2.0, 32: 2.0, 39: 0.0})
            elif row.question.type == "choice":  # positional ids, option "x" gets 0.8 wherever it is
                tables.append({20 + i: math.log(0.8 if o.key == "x" else 0.2) for i, o in enumerate(row.options)})
            else:
                key = row.question.instructions.split("Candidate: ", 1)[1].split(":", 1)[0]
                p = {"yes": YES[key], "no": 1 - YES[key]}
                tables.append({20 + i: math.log(p[o.key]) for i, o in enumerate(row.options)})
        return tables, len(prompts)


def test_mixed_request_keeps_reads_aligned():
    crit = {k: f"Option {k}" for k in YES}
    req = SystemOneIn(model="decision", state="s", permutations=2, questions={
        "pick": {"type": "choice", "instructions": "Which?", "criteria": {"x": "X", "y": "Y"}},
        "head": {"type": "labels", "instructions": "Tags?", "criteria": crit},
        "each": {"type": "labels", "instructions": "Tags?", "mode": "per_option", "criteria": crit},
    })
    engine = MixedEngine()
    out = engine.evaluate(req)
    assert out["answers"]["pick"]["choice"] == "x"
    assert out["answers"]["pick"]["probabilities"]["x"] == pytest.approx(0.8)
    assert "mode" not in out["answers"]["head"]
    assert out["answers"]["head"]["probabilities"]["a"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert out["answers"]["each"]["probabilities"] == pytest.approx(YES)
    assert len(engine.prompts) == 2 + 2 + 3 * 2


def test_many_permutations_read_each_option_twice():
    engine = PerOptionEngine()
    answer = engine.evaluate(body(permutations=7))["answers"]["actions"]
    assert len(engine.prompts) == 3 * 2
    assert answer["probabilities"] == pytest.approx(YES)


def test_raw_confidence_uses_temperature_one():
    engine = PerOptionEngine(temperatures={"1": {"noul": 4.0}})
    raw = {}
    engine._evaluate_once(body(), raw_confidence=raw)
    assert raw["actions"] == pytest.approx(0.9 * 0.8 * 0.6)


def test_long_per_option_expansion_is_rejected_before_reading():
    # 250 options of 1500 characters fit the 400k question limit, but repeating the listing per option would not.
    criteria = {f"o{i}": "x" * 1500 for i in range(250)}
    req = SystemOneIn(model="decision", state="s", questions={
        "actions": {"type": "labels", "instructions": "Pick", "mode": "per_option", "criteria": criteria}})
    engine = PerOptionEngine()
    with pytest.raises(RequestError) as err:
        engine.evaluate(req)
    assert err.value.loc == ["body", "questions", "actions"] and "per-option" in err.value.msg
    assert engine.prompts == []


def test_remote_engine_usage_counts_one_read_per_option(monkeypatch):
    from bielik_one.api.remote_vllm import RemoteVllmEngine

    monkeypatch.setattr(LetterEngine, "_evaluate_once", lambda self, body, **kw: {"usage": {"input_tokens": 0, "output_tokens": -1}})
    engine = RemoteVllmEngine.__new__(RemoteVllmEngine)
    engine.base_row_types = frozenset()
    for perms, expected in ((1, 3), (2, 6), (16, 6)):
        out = RemoteVllmEngine._evaluate_once(engine, body(permutations=perms))
        assert out["usage"]["output_tokens"] == expected
