"""Read-twice copies preserve answer-symbol meanings under permutations."""
from unittest.mock import patch

import pytest

from bielik_one.api.server import QuestionIn, question_to_row
from bielik_one.prompting import inference
from bielik_one.prompting.inference import inference_messages, inference_ids
from bielik_one.prompting.orders import _reordered


class CharacterTokenizer:
    """Deterministic tokenizer for prefix tests, including chat formatting."""
    all_special_tokens = []

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return list(map(ord, text))


def rows(kind="choice"):
    criteria = {
        "choice": {"x": "Apple", "y": "Pear"},
        "labels": {"x": "Apple", "y": "Pear"},
        "noul": {"false": "Incorrect", "true": "Correct"},
        "score": ["Poor", "Good"],
    }[kind]
    row, _ = question_to_row("Shared evidence", "q", QuestionIn(
        type=kind, instructions="Pick one", criteria=criteria,
    ))
    return row, _reordered(row, [1, 0])


def copies(text):
    first, rest = text.split("\n\nRead the input again:\n")
    return first, rest.rsplit("\n\n", 1)[0]


@pytest.mark.parametrize("kind,semantic", [
    ("choice", False), ("choice", True), ("labels", False), ("labels", True),
    ("noul", False), ("score", False),
])
def test_letter_copies_have_identical_mappings(kind, semantic):
    canonical, reverse = rows(kind)
    tok = CharacterTokenizer()
    for row in (canonical, reverse):
        text = inference_messages(
            tok, row, semantic, first_row=canonical, read_twice=True,
        )[-1]["content"]
        first, second = copies(text)
        assert first == second
        assert "A. " in first and "B. " in first
    if kind in ("choice", "labels"):
        first, _ = copies(text)
        assert "A. y: Pear" in first
        assert "B. x: Apple" in first


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_semantic_first_copy_identical_text_and_tokens_across_permutations(kind):
    canonical, reverse = rows(kind)
    tok = CharacterTokenizer()
    outputs = [inference_messages(
        tok, row, True, first_row=canonical, read_twice=True,
    )[-1]["content"] for row in (canonical, reverse)]
    first = [copies(text)[0] for text in outputs]
    assert first[0] == first[1]
    assert outputs[0] != outputs[1]
    symbols = ("no:", "yes:") if kind == "noul" else ("0:", "1:")
    first_options = first[1].split("Options:\n")[1].splitlines()
    second_options = copies(outputs[1])[1].split("Options:\n")[1].splitlines()
    assert first_options == list(reversed(second_options))
    assert all(symbol in first[1] for symbol in symbols)
    tokenized = [inference_ids(
        tok, row, True, first_row=canonical, read_twice=True,
    ) for row in (canonical, reverse)]
    decoded = "".join(map(chr, tokenized[0]))
    split = decoded.index("\n\nRead the input again:\n") + len("\n\nRead the input again:\n")
    assert tokenized[0][:split] == tokenized[1][:split]


@pytest.mark.parametrize("kind", ["choice", "labels", "noul", "score"])
def test_single_original_permutation_retains_previous_content_without_rerender(kind):
    canonical, _ = rows(kind)
    tok = CharacterTokenizer()
    standard = inference_messages(tok, canonical, True)
    content, tail = standard[-1]["content"].rsplit("\n\n", 1)
    expected = content + "\n\nRead the input again:\n" + content + "\n\n" + tail
    # An equal-but-distinct row occurs for the identity permutation in the API.
    identity = _reordered(canonical, [0, 1])
    with patch.object(inference, "messages", wraps=inference.messages) as render:
        fixed = inference_messages(tok, identity, True, first_row=canonical, read_twice=True)
    assert fixed[-1]["content"] == expected
    assert render.call_count == 1


@pytest.mark.parametrize("kind", ["choice", "labels", "noul", "score"])
def test_preview_and_analysis_keep_symbol_policy(kind):
    canonical, reverse = rows(kind)
    tok = CharacterTokenizer()
    outputs = [inference_messages(
        tok, row, True, first_row=canonical, questions=[canonical],
        questions_before_state=True, read_twice=True, analysis="Analysis suffix",
    )[-1]["content"] for row in (canonical, reverse)]
    assert all(text.count("Analysis suffix") == 1 for text in outputs)
    prefixes = [text.split("\n\nRead the input again:\n")[0] for text in outputs]
    if kind in ("noul", "score"):
        assert prefixes[0] == prefixes[1]
    else:
        assert prefixes[0] != prefixes[1]
        first, remainder = outputs[1].split("\n\nRead the input again:\n")
        second = remainder.split("\n\nAuxiliary analysis")[0]
        assert first == second
