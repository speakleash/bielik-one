"""Actual API prompt selection and inference metadata without GPU weights."""
import concurrent.futures
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from bielik_one.api.remote_vllm import _prompt_variants
from bielik_one.api.server import (
    DEFAULT_READ_TWICE_MAX_INPUT_TOKENS,
    RequestError,
    SystemOneIn,
)


def body(**kwargs):
    return SystemOneIn(
        state="state",
        model="decision",
        questions={"q": {"type": "noul", "instructions": "Is this correct?"}},
        **kwargs,
    )


def test_schema_defaults_and_explicit_disable():
    assert body().read_twice is True
    assert body().read_twice_max_input_tokens == DEFAULT_READ_TWICE_MAX_INPUT_TOKENS
    assert body(read_twice=False).read_twice is False
    assert body(read_twice_max_input_tokens=None).read_twice_max_input_tokens is None
    for value in (-1, True, 1.5, "4096"):
        with pytest.raises(ValidationError):
            body(read_twice_max_input_tokens=value)


@pytest.mark.parametrize("standard,repeated,context,threshold,enabled,expected", [
    (4096, 8200, 10000, 4096, True, 8200),
    (4097, 8202, 10000, 4096, True, 4097),
    (100, 199, 200, None, True, 199),
    (100, 200, 200, None, True, 100),
    (100, 199, 200, 4096, False, 100),
    (5000, 10000, 12000, None, True, 10000),
    (100, 199, 200, 0, True, 100),
])
def test_actual_backend_prompt(
    engine_factory, standard, repeated, context, threshold, enabled, expected
):
    fixture = engine_factory(context)
    row = SimpleNamespace(id="q", question=SimpleNamespace(type="noul"), options=[0, 1])

    def prompt(*args, **kwargs):
        return [1] * (repeated if kwargs["read_twice"] else standard)

    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ):
        _, tokens = fixture.engine.distributions(
            [row], read_twice=enabled, read_twice_max_input_tokens=threshold
        )
    assert len(fixture.requests) == 1
    assert len(fixture.requests[0]["prompt"]) == expected
    assert tokens == expected


def test_standard_overflow_never_calls_backend(engine_factory):
    fixture = engine_factory(100)
    row = SimpleNamespace(id="q", question=SimpleNamespace(type="noul"), options=[0, 1])
    with patch("bielik_one.api.remote_vllm.inference_ids", return_value=[1] * 100):
        with pytest.raises(RequestError):
            fixture.engine.distributions([row], read_twice=True)
    assert fixture.requests == []


def test_request_propagates_threshold(engine_factory):
    engine = engine_factory().engine
    seen = {}

    def distributions(prompts, **kwargs):
        seen.update(kwargs)
        return [{1000: math.log(0.5), 1001: math.log(0.5)}], 10

    engine.distributions = distributions
    with patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        engine.evaluate(body(read_twice_max_input_tokens=1234))
    assert seen["read_twice"] is True
    assert seen["read_twice_max_input_tokens"] == 1234


@pytest.mark.parametrize("variant", ["mixed", "threshold", "context", "disabled"])
def test_permutation_metadata(engine_factory, variant):
    fixture = engine_factory(200)
    counters = {False: 0, True: 0}

    def prompt(*args, **kwargs):
        repeated = kwargs["read_twice"]
        index = counters[repeated]
        counters[repeated] += 1
        if not repeated:
            return [1] * (100 + index if variant == "threshold" else 100)
        return [1] * (200 if variant == "context" or (variant == "mixed" and index) else 199)

    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        result = fixture.engine.evaluate(body(
            permutations=2,
            read_twice=variant != "disabled",
            read_twice_max_input_tokens=100,
            prompt_details=True,
        ))
    info = result["answers"]["q"]["prompt"]
    expected = {key: "standard" for key in ("mixed", "threshold", "context", "disabled")}
    reasons = {"mixed": ["context_limit"] * 2, "threshold": ["input_threshold"] * 2,
               "context": ["context_limit"] * 2, "disabled": ["disabled"] * 2}
    assert info["variant"] == expected[variant]
    assert [read["reason"] for read in info["reads"]] == reasons[variant]
    assert info["read_twice_reads"] + info["standard_reads"] == 2
    assert info["requested_read_twice"] is (variant != "disabled")
    assert _prompt_variants.get() is None


def test_per_option_labels_metadata(engine_factory):
    fixture = engine_factory()
    request = SystemOneIn(
        model="decision", state="state", permutations=2, prompt_details=True,
        questions={"tags": {"type": "labels", "mode": "per_option",
                            "instructions": "Select tags", "criteria": {"x": "X", "y": "Y"}}},
    )
    with patch("bielik_one.api.remote_vllm.inference_ids", return_value=[1] * 100), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        result = fixture.engine.evaluate(request)
    info = result["answers"]["tags"]["prompt"]
    assert info["variant"] == "read_twice"
    assert info["read_twice_reads"] == 4
    assert len(info["reads"]) == 4


@pytest.mark.parametrize("preview,analysis", [(True, ""), (False, "auxiliary"), (True, "auxiliary")])
def test_threshold_includes_preview_and_analysis(engine_factory, preview, analysis):
    fixture = engine_factory()
    row = SimpleNamespace(id="q", question=SimpleNamespace(type="noul"), options=[0, 1])
    calls = []

    def prompt(*args, **kwargs):
        calls.append(kwargs)
        return [1] * (100 + 10 * kwargs["questions_before_state"] + 10 * bool(kwargs["analysis"]))

    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ):
        fixture.engine.distributions(
            [row], questions=[row], questions_before_state=preview,
            analyses={"q": analysis}, read_twice=True, read_twice_max_input_tokens=100,
        )
    assert len(calls) == 1
    assert calls[0]["read_twice"] is False
    assert calls[0]["questions"] == [row]
    assert calls[0]["analysis"] == analysis


def test_metadata_is_request_local_after_error_and_between_threads(engine_factory):
    fixture = engine_factory(200)

    def prompt(*args, **kwargs):
        return [1] * (200 if kwargs["read_twice"] else 100)

    with patch("bielik_one.api.remote_vllm.inference_ids", return_value=[1] * 200):
        with pytest.raises(RequestError):
            fixture.engine.evaluate(body())
    assert _prompt_variants.get() is None
    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(fixture.engine.evaluate, [body(prompt_details=True), body(read_twice=False, prompt_details=True)]))
    assert results[0]["answers"]["q"]["prompt"]["reads"][0]["reason"] == "context_limit"
    assert results[1]["answers"]["q"]["prompt"]["reads"][0]["reason"] == "disabled"
    assert _prompt_variants.get() is None


def test_compact_metadata_and_opt_in_details(engine_factory):
    fixture = engine_factory()
    with patch("bielik_one.api.remote_vllm.inference_ids", return_value=[1] * 100), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        compact = fixture.engine.evaluate(body())
        detailed = fixture.engine.evaluate(body(prompt_details=True))
    info = compact["answers"]["q"]["prompt"]
    assert "reads" not in info
    assert "inference" not in compact["answers"]["q"]
    assert info["variant"] == "read_twice"
    assert info["read_twice_max_input_tokens"] == DEFAULT_READ_TWICE_MAX_INPUT_TOKENS
    assert info["max_context_tokens"] == fixture.engine.max_model_len
    assert len(detailed["answers"]["q"]["prompt"]["reads"]) == 1


def test_one_overflowing_per_option_read_disables_whole_question(engine_factory):
    fixture = engine_factory(200)
    count = 0

    def prompt(*args, **kwargs):
        nonlocal count
        if not kwargs["read_twice"]:
            return [1] * 100
        count += 1
        return [1] * (200 if count == 4 else 199)

    request = SystemOneIn(
        model="decision", state="state", permutations=2, prompt_details=True,
        questions={"tags": {"type": "labels", "mode": "per_option",
                            "instructions": "Pick tags", "criteria": {"x": "X", "y": "Y"}}},
    )
    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        result = fixture.engine.evaluate(request)
    info = result["answers"]["tags"]["prompt"]
    assert info["variant"] == "standard"
    assert info["reason"] == "context_limit"
    assert info["standard_reads"] == 4
    assert all(len(request["prompt"]) == 100 for request in fixture.requests)


def test_original_order_forwarded_to_every_permutation(engine_factory):
    fixture = engine_factory()
    observed = []

    def prompt(tok, row, semantic, **kwargs):
        if kwargs["read_twice"]:
            observed.append(([o.key for o in kwargs["first_row"].options],
                             [o.key for o in row.options]))
        return [1] * 100

    request = SystemOneIn(
        model="decision", state="state", permutations=2,
        questions={"q": {"type": "choice", "instructions": "Pick one",
                         "criteria": {"x": "X", "y": "Y"}}},
    )
    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        fixture.engine.evaluate(request)
    assert len(observed) == 2
    assert all(first == ["x", "y"] for first, second in observed)
    assert {tuple(second) for first, second in observed} == {("x", "y"), ("y", "x")}


def test_variants_are_independent_between_questions(engine_factory):
    fixture = engine_factory()
    request = SystemOneIn(
        model="decision", state="state", read_twice_max_input_tokens=100,
        questions={name: {"type": "noul", "instructions": name} for name in ("short", "long")},
    )

    def prompt(tok, row, semantic, **kwargs):
        return [1] * (100 if row.id == "short" else 101)

    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=prompt), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", return_value=[1000, 1001]
    ), patch("bielik_one.api.server.answer_token_ids", return_value=[1000, 1001]):
        result = fixture.engine.evaluate(request)
    assert result["answers"]["short"]["prompt"]["variant"] == "read_twice"
    assert result["answers"]["long"]["prompt"]["variant"] == "standard"
