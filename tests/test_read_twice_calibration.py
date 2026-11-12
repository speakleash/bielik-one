"""The shipped one-order temperatures apply to every request mode; a request temperature still wins."""

import math

import pytest

from bielik_one.api import server
from bielik_one.api.server import LetterEngine, SystemOneIn


class ChoiceEngine(LetterEngine):
    model_name = "decision"
    capacity = 255
    labels_threshold = 0.5
    semantic = True
    parking = None
    tok = None
    temperatures = {"1": {"choice": 2.0}}

    def distributions(self, prompts, **kwargs):
        self.kwargs = kwargs
        # Option "x" gets 0.8 wherever it is placed.
        return [{20 + i: math.log(0.8 if o.key == "x" else 0.2) for i, o in enumerate(row.options)} for row in prompts], 1


@pytest.fixture(autouse=True)
def positional_tokens(monkeypatch):
    monkeypatch.setattr(server, "answer_token_ids", lambda tok, row, semantic: [20 + i for i in range(len(row.options))])


def probability_of_x(**options):
    body = SystemOneIn(model="decision", state="s", questions={
        "q": {"type": "choice", "instructions": "Which?", "criteria": {"x": "X", "y": "Y"}}}, **options)
    return ChoiceEngine().evaluate(body)["answers"]["q"]["probabilities"]["x"]


CALIBRATED = math.sqrt(0.8) / (math.sqrt(0.8) + math.sqrt(0.2))  # softmax(log p / 2)


def test_default_request_is_calibrated():
    assert probability_of_x() == pytest.approx(CALIBRATED)


def test_read_twice_is_calibrated():
    assert probability_of_x(read_twice=True) == pytest.approx(CALIBRATED)


def test_questions_before_state_is_calibrated():
    assert probability_of_x(questions_before_state=True) == pytest.approx(CALIBRATED)


@pytest.mark.parametrize("orders", [2, 4])
def test_more_option_orders_use_the_one_order_table(orders):
    assert probability_of_x(permutations=orders) == pytest.approx(CALIBRATED)


def test_a_two_order_table_wins_when_present(monkeypatch):
    monkeypatch.setattr(ChoiceEngine, "temperatures", {"1": {"choice": 2.0}, "2": {"choice": 1.0}})
    assert probability_of_x(permutations=2) == pytest.approx(0.8)
    assert probability_of_x() == pytest.approx(CALIBRATED)


def test_request_temperature_still_wins():
    assert probability_of_x(read_twice=True, temperature=1) == pytest.approx(0.8)
