"""Read multilabel probabilities from served logprobs.

The labels head occupies token ids that prompts never contain, plus one anchor with a zero projection.
The backend returns a softmax over that finite set; z_i = log p_i - log p_anchor (+ bias) and each label
probability is sigmoid(z_i). Exponentiating the logprobs directly would be wrong.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def sigmoid_from_logprobs(
    table: Mapping,
    option_ids: Sequence[int],
    anchor_id: int,
    temperature: float = 1.0,
) -> list[float]:
    """Recover sigmoid(z) from log p, subtracting the anchor. Does not exponentiate first."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    def logprob(token_id: int) -> float:
        entry = table.get(int(token_id))
        if entry is None:
            entry = table.get(str(int(token_id)))
        if entry is None:
            raise RuntimeError(
                f"token {token_id} is missing from the logprobs; "
                f"max_logprobs must cover the options plus the anchor"
            )
        if hasattr(entry, "logprob"):
            return float(entry.logprob)
        if isinstance(entry, Mapping):
            return float(entry["logprob"])
        return float(entry)

    anchor = logprob(anchor_id)
    return [
        stable_sigmoid((logprob(token_id) - anchor) / temperature)
        for token_id in option_ids
    ]
