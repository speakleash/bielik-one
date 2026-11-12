"""Prompt rendering and answer-token ids for the decision heads (the training prompt format)."""

from __future__ import annotations

import string
import weakref


from bielik_one.prompting.schema import Row, span_text

MAX_LETTERS = 26
MAX_LABELS = 255
SYSTEM = (
    "You are a decision model. Read the state and the question, then answer "
    "with the letter of the single best option. Use only the information in the state."
)
LABELS_SYSTEM = (
    "You are a decision model. Read the state and the question, then answer "
    "with the letters of every option that applies. Use only the information in the state."
)
SINGLE_TAIL = "Answer with the letter of the correct option."
LABELS_TAIL = "Answer with the letters of all options that apply."
NOUL_SYSTEM = (
    "You are a decision model. Read the state and the question, then answer "
    "yes or no. Use only the information in the state."
)
SCORE_SYSTEM = (
    "You are a decision model. Read the state and the question, then answer "
    "with the score digit. Use only the information in the state."
)
NOUL_TAIL = "Answer yes or no."
SCORE_TAIL = "Answer with the score digit."
SEMANTIC_TYPES = ("noul", "score")


def _one_token_after_newline(tok, label: str) -> bool:
    """The label is one token after a newline, and stays that token in a line that starts with it."""
    newline = len(tok.encode("\n", add_special_tokens=False))
    alone = tok.encode("\n" + label, add_special_tokens=False)
    inline = tok.encode("\n" + label + ". x: y", add_special_tokens=False)
    return len(alone) == newline + 1 and inline[newline] == alone[-1]


_POOLS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_SPECIALS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
WORD_JOINER = "\u2060"


def special_strings(tok) -> tuple[str, ...]:
    """The tokenizer's control-token strings, longest first: special tokens, plus added tokens that look like
    tags (`<think>`, `</think>`, `<tool_call>`, `<|fim_prefix|>`), which some tokenizers do not mark special."""
    try:
        return _SPECIALS[tok]
    except (KeyError, TypeError):
        pass
    found = {str(s) for s in (getattr(tok, "all_special_tokens", None) or ())}
    for token in (getattr(tok, "added_tokens_decoder", None) or {}).values():
        content = str(getattr(token, "content", token))
        if getattr(token, "special", False) or (
            len(content) > 2 and content.startswith("<") and content.endswith(">")
        ):
            found.add(content)
    out = tuple(sorted((s for s in found if len(s) > 1), key=len, reverse=True))
    try:
        _SPECIALS[tok] = out
    except TypeError:
        pass
    return out


def neutralize(tok, text: str) -> str:
    """Client text that spells a control token (e.g. `<|im_end|>`) is kept visible but can no longer be
    tokenized as that token: a word joiner goes after its first character. Text without such strings,
    including every training and benchmark prompt, is unchanged."""
    for special in special_strings(tok):
        if special in text:
            text = text.replace(special, special[0] + WORD_JOINER + special[1:])
    return text


def safe_messages(tok, chat: list[dict]) -> list[dict]:
    """Neutralize every non-system message; system prompts are ours."""
    return [
        {**m, "content": neutralize(tok, m["content"])}
        if m.get("role") != "system" and isinstance(m.get("content"), str)
        else m
        for m in chat
    ]


def label_pool(tok) -> list[str]:
    """Every label the tokenizer can read out, in option order: A..Z, a..z, then AA, AB, ..."""
    if tok not in _POOLS:
        upper = string.ascii_uppercase
        pool = [
            c
            for c in upper + string.ascii_lowercase
            if _one_token_after_newline(tok, c)
        ]
        pool += [
            a + b for a in upper for b in upper if _one_token_after_newline(tok, a + b)
        ]
        if pool[:MAX_LETTERS] != list(upper):
            raise ValueError("the bare letters A..Z must be single tokens")
        _POOLS[tok] = pool
    return _POOLS[tok]


def label_names(tok, n: int) -> list[str]:
    """Labels for `n` options. Up to 26 are the plain letters and need no tokenizer lookup."""
    if n < 1 or n > MAX_LABELS:
        raise ValueError(f"label readout covers 1..{MAX_LABELS} options")
    if n <= MAX_LETTERS:
        return [chr(65 + i) for i in range(n)]
    pool = label_pool(tok)
    if len(pool) < n:
        raise ValueError(f"tokenizer has {len(pool)} one-token labels, need {n}")
    return pool[:n]


def semantic_labels(row: Row) -> list[str]:
    """Answer symbols that travel with the option: `no`/`yes`, or the score digit."""
    if row.question.type == "noul":
        labels = [option.key for option in row.options]
        if any(label not in ("no", "yes") for label in labels):
            raise ValueError(f"semantic noul options are no and yes, got {labels}")
        return labels
    if row.question.type == "score":
        labels = [option.key for option in row.options]
        if any(len(label) != 1 or label not in "0123456789" for label in labels):
            raise ValueError(f"semantic score keys are the digits 0-9, got {labels}")
        return labels
    raise ValueError(f"{row.question.type} keeps letter labels")


def render_user(
    row: Row, labels: list[str] | None = None, semantic: bool = False
) -> str:
    n = len(row.options)
    if semantic and row.question.type in SEMANTIC_TYPES:
        labels = semantic_labels(row)
        lines = [
            f"{labels[i]}: {span_text(option)}" for i, option in enumerate(row.options)
        ]
        tail = NOUL_TAIL if row.question.type == "noul" else SCORE_TAIL
    else:
        if labels is None:
            if n > MAX_LETTERS:
                raise ValueError(
                    "more than 26 options need labels from label_names(tok, n)"
                )
            labels = [chr(65 + i) for i in range(n)]
        lines = [
            f"{labels[i]}. {option.key}: {span_text(option)}"
            for i, option in enumerate(row.options)
        ]
        tail = LABELS_TAIL if row.question.type == "labels" else SINGLE_TAIL
    state = row.state.strip()
    context = (
        f"State:\n{row.state}\n\n"
        if state and state != row.question.instructions.strip()
        else ""
    )
    return (
        f"{context}Question: {row.question.instructions}\n\n"
        f"Options:\n" + "\n".join(lines) + "\n\n" + tail
    )


def messages(
    row: Row, labels: list[str] | None = None, semantic: bool = False
) -> list[dict]:
    if semantic and row.question.type == "noul":
        system = NOUL_SYSTEM
    elif semantic and row.question.type == "score":
        system = SCORE_SYSTEM
    elif row.question.type == "labels":
        system = LABELS_SYSTEM
    else:
        system = SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_user(row, labels, semantic)},
    ]


def bare_letter_ids(tok, n: int) -> list[int]:
    """Ids of the first `n` labels as the model emits them after a newline, not the spaced `▁A` tokens."""
    ids = []
    for label in label_names(tok, n):
        encoded = tok.encode("\n" + label, add_special_tokens=False)
        ids.append(int(encoded[-1]))
    if len(set(ids)) != n:
        raise ValueError("label tokens are not unique")
    return ids


def prompt_ids(tok, row: Row, semantic: bool = False) -> list[int]:
    if row.metadata.get("training_prompt"):
        from bielik_one.prompting.inference import training_prompt_ids

        return training_prompt_ids(tok, row, semantic)
    labels = None
    if not (semantic and row.question.type in SEMANTIC_TYPES):
        labels = label_names(tok, len(row.options))
    rendered = tok.apply_chat_template(
        safe_messages(tok, messages(row, labels, semantic)),
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = rendered["input_ids"] if not isinstance(rendered, list) else rendered
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def symbol_token_id(tok, label: str) -> int:
    """The token id of `label` after a newline, the same id the option line starts with."""
    newline = tok.encode("\n", add_special_tokens=False)
    alone = tok.encode("\n" + label, add_special_tokens=False)
    inline = tok.encode("\n" + label + ": x", add_special_tokens=False)
    if len(alone) != len(newline) + 1 or inline[len(newline)] != alone[-1]:
        raise ValueError(f"{label!r} is not one token at the answer position")
    return int(alone[-1])


def semantic_token_ids(tok) -> dict:
    """`no`, `yes`, and the digits 0-9, each one token after a newline."""
    digits = [symbol_token_id(tok, str(i)) for i in range(10)]
    ids = {
        "no": symbol_token_id(tok, "no"),
        "yes": symbol_token_id(tok, "yes"),
        "digits": digits,
    }
    if len({ids["no"], ids["yes"], *digits}) != 12:
        raise ValueError("semantic answer tokens are not unique")
    return ids


def answer_token_ids(tok, row: Row, semantic: bool = False) -> list[int]:
    """Token ids lined up with `row.options`. Choice stays on letters."""
    if semantic and row.question.type in SEMANTIC_TYPES:
        return [symbol_token_id(tok, label) for label in semantic_labels(row)]
    return bare_letter_ids(tok, len(row.options))
