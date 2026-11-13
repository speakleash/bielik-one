"""Prompt text invariants underlying shared prefix caching and answer permutation."""
from bielik_one.api.server import QuestionIn, question_to_row
from bielik_one.prompting.inference import inference_messages, inference_ids
from bielik_one.prompting.orders import _reordered


class CharacterTokenizer:
    """Deterministic tokenizer for exact prefix tests, including chat formatting."""
    all_special_tokens = []

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return list(map(ord, text))


def rows():
    row, _ = question_to_row("Shared evidence", "q", QuestionIn(
        type="choice", instructions="Pick one", criteria={"x": "Apple", "y": "Pear"},
    ))
    return row, _reordered(row, [1, 0])


def test_first_copy_identical_text_and_tokens_across_permutations():
    canonical, reverse = rows()
    tok = CharacterTokenizer()
    outputs = [inference_messages(tok, row, first_row=canonical, read_twice=True)[-1]["content"]
               for row in (canonical, reverse)]
    first = [text.split("\n\nRead the input again:\n")[0] for text in outputs]
    assert first[0] == first[1]
    assert outputs[0] != outputs[1]
    assert outputs[0].index("Apple") < outputs[0].index("Pear")
    second_reverse = outputs[1].split("\n\nRead the input again:\n")[1]
    assert second_reverse.index("Pear") < second_reverse.index("Apple")
    tokenized = [inference_ids(tok, row, first_row=canonical, read_twice=True)
                 for row in (canonical, reverse)]
    # Prefix includes complete first copy and the separator, rather than just the state.
    prefix = "<user>" + first[0] + "\n\nRead the input again:\n"
    # System messages can precede the user message; locate the shared user prefix.
    decoded = ["".join(map(chr, ids)) for ids in tokenized]
    for text in decoded:
        assert prefix in text
    split = decoded[0].index("\n\nRead the input again:\n") + len("\n\nRead the input again:\n")
    assert tokenized[0][:split] == tokenized[1][:split]


def test_single_original_permutation_retains_previous_read_twice_content():
    canonical, _ = rows()
    tok = CharacterTokenizer()
    standard = inference_messages(tok, canonical)
    content, tail = standard[-1]["content"].rsplit("\n\n", 1)
    expected = content + "\n\nRead the input again:\n" + content + "\n\n" + tail
    fixed = inference_messages(tok, canonical, first_row=canonical, read_twice=True)
    assert fixed[-1]["content"] == expected
    assert fixed[0] == standard[0]


def test_preview_is_stable_and_analysis_remains_outside_repetition():
    canonical, reverse = rows()
    tok = CharacterTokenizer()
    outputs = [inference_messages(
        tok, row, first_row=canonical, questions=[canonical],
        questions_before_state=True, read_twice=True, analysis="Analysis suffix",
    )[-1]["content"] for row in (canonical, reverse)]
    assert outputs[0].split("Read the input again:")[0] == outputs[1].split("Read the input again:")[0]
    assert all(text.count("Analysis suffix") == 1 for text in outputs)
