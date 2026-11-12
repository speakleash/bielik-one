"""Finite, normalized answers for every accepted positive temperature."""

import math

import pytest
import torch

from bielik_one.api.server import (
    LetterEngine,
    SystemOneIn,
    apply_sigmoid_temperature,
    build_app,
)
from bielik_one.prompting.intervals import at_temperature


@pytest.mark.parametrize("temperature", [0.001, 1e-50, 5e-324, 1e50, 1e308])
def test_categorical_extreme_temperature(temperature):
    probs = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.0, 1.0]])
    result = at_temperature(probs, temperature)
    assert result.dtype == probs.dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.sum(dim=-1), torch.ones(3))
    assert (result >= 0).all() and (result <= 1).all()
    assert result[0, 1] >= result[0, 0]
    torch.testing.assert_close(result[1], torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("temperature", [0.001, 1e-50, 5e-324, 1e50, 1e308])
def test_labels_extreme_temperature(temperature):
    result = apply_sigmoid_temperature([0, 0.1, 0.5, 0.9, 1], temperature)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in result)
    assert result == sorted(result)
    assert result[2] == 0.5
    assert result[0] == pytest.approx(1 - result[-1])


@pytest.mark.parametrize("temperature", [0.25, 0.8, 1.0, 1.625, 2.878, 16.0])
def test_ordinary_temperatures_match_previous_formulas(temperature):
    probs = torch.tensor([0.1, 0.3, 0.6])
    old = torch.softmax(probs.log() / temperature, dim=-1)
    torch.testing.assert_close(at_temperature(probs, temperature), old)
    expected = [1 / (1 + math.exp(-math.log(p / (1 - p)) / temperature)) for p in probs.tolist()]
    assert apply_sigmoid_temperature(probs.tolist(), temperature) == pytest.approx(expected)


class LabelsEngine(LetterEngine):
    model_name = "decision"
    capacity = 255
    labels_threshold = 0.5
    semantic = True
    temperatures = {}
    parking = {"option_ids": [10, 11], "anchor_id": 12}

    def distributions(self, prompts, **kwargs):
        return [{10: -math.log(9), 11: math.log(9), 12: 0.0} for _ in prompts], len(prompts)


@pytest.mark.parametrize("temperature", [0.001, 1e-50, 5e-324])
def test_labels_http_response_is_serializable(temperature):
    from fastapi.testclient import TestClient

    body = SystemOneIn(
        model="decision", state="Example", temperature=temperature,
        questions={"tags": {"type": "labels", "instructions": "Pick tags",
                             "criteria": {"a": "First", "b": "Second"}}},
    )
    with TestClient(build_app(LabelsEngine())) as client:
        response = client.post("/v1/systemone", json=body.model_dump())
    assert response.status_code == 200
    answer = response.json()["answers"]["tags"]
    assert answer["selected"] == ["b"]
    assert answer["probabilities"] == {"a": 0.0, "b": 1.0}
