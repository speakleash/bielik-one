"""Per-row readout, independent of vLLM internals and testable on CPU."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import torch
from torch import nn
from torch.nn import functional as F


BASE_HEAD_FLAG = "systemone_head"
BASE_HEAD_VALUE = "base"


class StepHeads:
    """Head outputs of the most recent logits computation, for every sampled row, plus the LoRA row mapping.

    compute_logits and the sampler's logits processors run one after the other in the same worker step, so the
    processor reads exactly the rows the model just produced. Only populated when base-row heads are enabled.
    """

    def __init__(self):
        self.heads = None
        self.sampler_indices = None
        self.token_ids = None

    def set(self, heads, sampler_indices, token_ids):
        self.heads, self.sampler_indices, self.token_ids = heads, sampler_indices, token_ids


LAST_STEP = StepHeads()


def update_flags(flags: dict, batch_update) -> bool:
    """Track flagged batch rows with vLLM's BatchUpdate contract: removed, then added, then moved/swapped."""
    if batch_update is None:
        return False
    changed = False
    for index in batch_update.removed:
        changed |= flags.pop(index, None) is not None
    for index, params, _prompt, _output in batch_update.added:
        extra = getattr(params, "extra_args", None) or {}
        if extra.get(BASE_HEAD_FLAG) == BASE_HEAD_VALUE:
            flags[index] = True
            changed = True
        elif flags.pop(index, None) is not None:
            changed = True
    for a, b, direction in batch_update.moved:
        fa, fb = flags.pop(a, None), flags.pop(b, None)
        if fa is not None:
            flags[b] = fa
        if fb is not None and getattr(direction, "name", str(direction)) == "SWAP":
            flags[a] = fb
        changed |= fa is not None or fb is not None
    return changed


def apply_base_heads(logits: torch.Tensor, rows: list[int], step: StepHeads) -> torch.Tensor:
    """Replace the logits of flagged base-model rows by decision-head outputs; other rows stay untouched."""
    if not rows:
        return logits
    if step.heads is None:
        raise RuntimeError("Base-row decision heads requested, but this server does not enable them")
    if step.heads.shape[0] != logits.shape[0]:
        raise RuntimeError("Decision-head rows do not match the sampler batch")
    index = torch.tensor(rows, device=logits.device, dtype=torch.long)
    # Only rows without a LoRA adapter qualify. A flagged LoRA row (possible from a hand-written request) already
    # carries head-only logits from selective(), so it is left untouched; no device-side assertion that could
    # bring down the engine for every user.
    base = step.sampler_indices.index_select(0, index) == -1
    values = step.heads.index_select(0, index).to(logits.dtype)
    rows_now = logits.index_select(0, index)
    replaced = torch.full_like(rows_now, -torch.inf)
    replaced[:, step.token_ids] = values
    logits.index_copy_(0, index, torch.where(base.unsqueeze(1), replaced, rows_now))
    return logits


class DecisionReadout(nn.Module):
    def __init__(self, weight: torch.Tensor, token_ids: torch.Tensor, *,
                 vocab_size: int, anchor_id: int, base_row_heads: bool = False):
        super().__init__()
        if weight.ndim != 2 or token_ids.ndim != 1 or weight.shape[0] != token_ids.numel():
            raise ValueError("Head weights and token IDs must have matching rows")
        if token_ids.dtype != torch.int64 or not torch.isfinite(weight).all():
            raise ValueError("Expected int64 IDs and finite head weights")
        ids = token_ids.tolist()
        if len(set(ids)) != len(ids) or any(i < 0 or i >= vocab_size for i in ids):
            raise ValueError("Head IDs must be distinct and within the original vocabulary")
        if anchor_id not in ids or torch.count_nonzero(weight[ids.index(anchor_id)]):
            raise ValueError("Anchor must be present and have an exactly zero projection")
        self.vocab_size = vocab_size
        self.base_row_heads = base_row_heads
        self.register_buffer("weight", weight.detach().float().clone(), persistent=False)
        self.register_buffer("token_ids", token_ids.detach().clone(), persistent=False)

    def forward(self, hidden: torch.Tensor, logits: torch.Tensor,
                sampler_indices: torch.Tensor) -> torch.Tensor:
        if (hidden.ndim != 2 or logits.ndim != 2 or sampler_indices.ndim != 1
                or hidden.shape[0] != logits.shape[0]
                or sampler_indices.numel() != hidden.shape[0]
                or hidden.shape[1] != self.weight.shape[1]
                or logits.shape[1] != self.vocab_size):
            raise ValueError("Logits rows, hidden states and LoRA sampler mapping do not align")
        if sampler_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("LoRA sampler mapping must contain integer slot indices")
        # TP=1, max_loras=1: -1 is native base, slot 0 is the fixed decision adapter.
        # Async assertion avoids a host synchronization on every GPU batch.
        torch._assert_async(
            ((sampler_indices == -1) | (sampler_indices == 0)).all(),
            "Only the fixed decision LoRA in slot0 and native base are supported",
        )
        all_heads = F.linear(hidden.float(), self.weight)
        if self.base_row_heads:
            LAST_STEP.set(all_heads, sampler_indices, self.token_ids)
        decision_logits = all_heads.to(logits.dtype)
        previous = logits.index_select(1, self.token_ids)
        selected = torch.where((sampler_indices == 0).unsqueeze(1), decision_logits, previous)
        return logits.index_copy_(1, self.token_ids, selected)

    def selective(self, hidden: torch.Tensor, sampler_indices: torch.Tensor,
                  native_projection) -> torch.Tensor:
        """Project native rows only; decision rows have finite head columns only.

        Logits computation is outside the compiled backbone. GPU nonzero partitions
        sampler rows; its synchronization cost is included in serving benchmarks.
        The standard sampler still needs a dense vocabulary-shaped output buffer.
        """
        if (hidden.ndim != 2 or sampler_indices.ndim != 1
                or sampler_indices.numel() != hidden.shape[0]
                or hidden.shape[1] != self.weight.shape[1]
                or sampler_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Hidden states and sampler mapping do not align")
        torch._assert_async(
            ((sampler_indices == -1) | (sampler_indices == 0)).all(),
            "Only the fixed decision LoRA in slot0 and native base are supported",
        )
        if self.base_row_heads:
            # Heads for every row (rows x 523, negligible); the logits processor uses them only for flagged rows.
            all_heads = F.linear(hidden.float(), self.weight)
            LAST_STEP.set(all_heads, sampler_indices, self.token_ids)
        base_rows = (sampler_indices == -1).nonzero().flatten()
        decision_rows = (sampler_indices == 0).nonzero().flatten()
        # vLLM's native logits processor produces FP32 even for BF16 hidden states.
        result = torch.full((hidden.shape[0], self.vocab_size), -torch.inf,
                            dtype=torch.float32, device=hidden.device)
        if base_rows.numel():
            native = native_projection(hidden.index_select(0, base_rows))
            if native is None or native.shape != (base_rows.numel(), self.vocab_size):
                raise ValueError("Native projection returned an unexpected shape")
            result.index_copy_(0, base_rows, native.float())
        if decision_rows.numel():
            heads = (all_heads.index_select(0, decision_rows) if self.base_row_heads
                     else F.linear(hidden.index_select(0, decision_rows).float(), self.weight))
            result[decision_rows[:, None], self.token_ids[None, :]] = heads
        return result


def package_file(directory: Path, name) -> Path:
    """A file named by a manifest, inside `directory`. Only a bare relative name is accepted; symlinks are fine
    (HF cache snapshots link into blobs/), and the caller's sha256 check pins the content."""
    if not isinstance(name, str) or not name:
        raise ValueError("weights_file must be a file name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or "\\" in name or any(part in ("..", "") for part in pure.parts):
        raise ValueError("weights_file must stay inside the package directory")
    return Path(directory) / pure
