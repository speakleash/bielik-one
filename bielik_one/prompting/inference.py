"""Inference prompt variants; the default remains byte-for-byte training compatible."""

from __future__ import annotations

import json
from types import SimpleNamespace
from collections.abc import Sequence

from bielik_one.prompting.letter import (
    SEMANTIC_TYPES,
    label_names,
    messages,
    neutralize,
    prompt_ids,
    safe_messages,
)
from bielik_one.prompting.schema import Row


def inference_messages(
    tok,
    row: Row,
    semantic: bool = False,
    *,
    questions: Sequence[Row] = (),
    questions_before_state: bool = False,
    read_twice: bool = False,
    analysis: str = "",
    header_style: str = "full",
) -> list[dict]:
    labels = (
        None
        if semantic and row.question.type in SEMANTIC_TYPES
        else label_names(tok, len(row.options))
    )
    result = messages(row, labels, semantic)
    if not (questions_before_state or read_twice or analysis):
        return result
    content, tail = result[-1]["content"].rsplit("\n\n", 1)
    if questions_before_state:
        # Stable option keys, rather than positional letters, keep this shared
        # prefix identical across all permutations of the target question.
        preview = [
            {
                "id": item.id,
                "type": item.question.type,
                "instructions": item.question.instructions,
                "options": {option.key: option.text for option in item.options},
            }
            for item in (questions or [row])
        ]
        if header_style == "text":
            content = (
                "Questions that will be asked about the context:\n"
                + "\n".join(
                    "- " + item.question.instructions for item in (questions or [row])
                )
                + "\n\n"
                + content
            )
        elif header_style == "full":
            content = (
                "Questions for this state:\n"
                + json.dumps(preview, ensure_ascii=False)
                + "\n\n"
                + content
            )
        else:
            raise ValueError(f"unknown question header style: {header_style}")
        content += "\n\nTarget question: " + json.dumps(row.id, ensure_ascii=False)
    if read_twice:
        content = content + "\n\nRead the input again:\n" + content
    if analysis:
        content += (
            "\n\nAuxiliary analysis (may contain errors; verify against the state):\n"
            + analysis
        )
    result[-1] = {**result[-1], "content": content + "\n\n" + tail}
    return result


def inference_ids(tok, row: Row, semantic: bool = False, **kwargs) -> list[int]:
    if not any(
        kwargs.get(key) for key in ("questions_before_state", "read_twice", "analysis")
    ):
        return prompt_ids(tok, row, semantic)
    rendered = tok.apply_chat_template(
        safe_messages(tok, inference_messages(tok, row, semantic, **kwargs)),
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = rendered["input_ids"] if not isinstance(rendered, list) else rendered
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def reasoning_messages(row: Row, tok=None) -> list[dict]:
    """No target or synthetic labels from the evaluation schema enter generation. With `tok`, client text is
    neutralized against that tokenizer's control tokens."""
    payload = {
        "state": row.state,
        "question": row.question.instructions,
        "type": row.question.type,
        "options": {option.key: option.text for option in row.options},
    }
    text = json.dumps(payload, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "Analyze the decision using only the supplied state. Give a concise explanation, "
                "check exceptions and missing information, and refer to options by their keys. "
                "Your analysis will be checked by a separate decision model."
            ),
        },
        {
            "role": "user",
            "content": (neutralize(tok, text) if tok is not None else text),
        },
    ]


def training_prompt_ids(tok, row: Row, semantic: bool = False) -> list[int]:
    """Render an explicit dataset variant identically to the inference path."""
    spec = row.metadata["training_prompt"]
    previews = [
        SimpleNamespace(
            id=p["id"],
            question=SimpleNamespace(type=p["type"], instructions=p["instructions"]),
            options=[SimpleNamespace(key=k, text=v) for k, v in p["options"].items()],
        )
        for p in spec.get("questions", [])
    ]
    return inference_ids(
        tok,
        row,
        semantic,
        questions=previews,
        questions_before_state=spec.get("questions_before_state", False),
        read_twice=spec.get("read_twice", False),
    )


def short_think_ids(
    tok, row: Row, semantic: bool, analysis: str | None = None
) -> list[int]:
    """Assistant prefill, then an optional explicitly closed thought before the head."""
    rendered = tok.apply_chat_template(
        safe_messages(tok, inference_messages(tok, row, semantic)),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not rendered.rstrip().endswith("<think>"):
        rendered += "<think>\n"
    rendered += "Short reasoning:"
    if analysis is not None:
        rendered += neutralize(tok, analysis) + "\n</think>\n"
    return [int(i) for i in tok.encode(rendered, add_special_tokens=False)]
