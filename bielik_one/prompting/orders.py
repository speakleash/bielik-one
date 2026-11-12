"""Option reordering helpers for permutation averaging."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence


from bielik_one.prompting.schema import Row


def _reversed(row: Row) -> Row:
    return _reordered(row, list(reversed(range(len(row.options)))))


def _reordered(row: Row, perm: Sequence[int]) -> Row:
    options = [row.options[i] for i in perm]
    criteria = row.question.criteria
    if isinstance(criteria, list):
        reordered = [criteria[i] for i in perm]
    else:
        reordered = {option.key: option.text for option in options}
    return row.model_copy(
        update={
            "options": options,
            "target_probs": [row.target_probs[i] for i in perm],
            "question": row.question.model_copy(update={"criteria": reordered}),
        }
    )


def _alt_orders(k: int, n: int, rng: random.Random) -> list[tuple[int, ...]]:
    """Up to `n` permutations other than the identity. Fewer when `k! - 1` is smaller."""
    identity = tuple(range(k))
    if k < 2 or n < 1:
        return []
    if math.factorial(k) <= n + 1:
        return [perm for perm in itertools.permutations(identity) if perm != identity]
    seen = {identity}
    out = []
    while len(out) < n:
        perm = list(identity)
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out
