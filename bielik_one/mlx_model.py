"""The MLX side of bielik_one.mlx_backend: the base model with a switchable LoRA, and the decision heads."""

from __future__ import annotations

import json
import threading
from pathlib import Path, PurePosixPath

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from bielik_one.mlx_backend import head_logprobs, lora_pairs, lora_scale, module_index


class Switch:
    """Process-wide adapter switch. MLX is lazy: it is read while a graph is built, so it is set under the lock."""

    on = False


class SwitchedLoRALinear(nn.Module):
    """y = base(x) [+ scale * (x @ A) @ B while Switch.on]. The base Linear is the one already in the model."""

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = a  # (in, r)
        self.lora_b = b  # (r, out)
        self.scale = scale

    def __call__(self, x):
        y = self.base(x)
        if Switch.on:
            z = (x @ self.lora_a) @ self.lora_b
            y = y + (self.scale * z).astype(y.dtype)
        return y


def attach_lora(model: nn.Module, adapter_dir: Path, rank: int | None, dtype=mx.bfloat16) -> tuple[int, int]:
    """Wrap every LoRA-targeted Linear of `model` in a SwitchedLoRALinear. Returns (modules, adapter bytes)."""
    scale = lora_scale(json.loads((adapter_dir / "adapter_config.json").read_text()), rank)
    linears = {path: m for path, m in model.named_modules() if isinstance(m, nn.Linear)}
    index = module_index(linears)
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    replacements, nbytes = [], 0
    for suffix, ab in lora_pairs(weights).items():
        if suffix not in index:
            raise ValueError(f"adapter module {suffix} has no unique unquantized Linear in the MLX model")
        path = index[suffix]
        linear = linears[path]
        # PEFT stores lora_A (r, in) and lora_B (out, r): y += scale * x A^T B^T.
        a = weights[ab["A"]].T.astype(dtype)
        b = weights[ab["B"]].T.astype(dtype)
        if a.shape[0] != linear.weight.shape[1] or b.shape[1] != linear.weight.shape[0] or a.shape[1] != b.shape[0]:
            raise ValueError(f"{suffix}: adapter shapes {a.shape} x {b.shape} vs weight {linear.weight.shape}")
        mx.eval(a, b)
        nbytes += a.nbytes + b.nbytes
        replacements.append((path, SwitchedLoRALinear(linear, a, b, scale)))
    del weights
    model.update_modules(tree_unflatten(replacements))
    return len(replacements), nbytes


class Heads:
    """The decision heads (float32) of heads.json, checked as the vLLM plugin checks them."""

    def __init__(self, root: Path, hidden_size: int):
        manifest = json.loads((root / "heads.json").read_text())
        name = manifest["weights_file"]
        pure = PurePosixPath(name)
        if pure.is_absolute() or "\\" in name or ".." in pure.parts:
            raise ValueError("weights_file must stay inside the package directory")
        # The checksum itself is part of launch.verify (heads.json weights_sha256 vs SHA256SUMS.json).
        tensors = mx.load(str(root / pure))
        self.weight = tensors["weight"].astype(mx.float32)  # (rows, hidden)
        ids = [int(i) for i in tensors["token_ids"].tolist()]
        if self.weight.ndim != 2 or self.weight.shape != (len(ids), hidden_size):
            raise ValueError("head weights do not match the token ids and the model's hidden size")
        if len(set(ids)) != len(ids) or not mx.all(mx.isfinite(self.weight)).item():
            raise ValueError("head ids must be distinct and head weights finite")
        self.column = {t: i for i, t in enumerate(ids)}
        anchor = int(manifest["anchor_id"])
        if anchor not in self.column or mx.any(self.weight[self.column[anchor]] != 0).item():
            raise ValueError("the anchor must be a head with an exactly zero projection")
        mx.eval(self.weight)


class MlxRunner:
    """The base model loaded once, with the switchable LoRA and the heads; one graph on the GPU at a time."""

    def __init__(self, base: Path, root: Path, info: dict, cache_limit_gb: float = 4.0):
        mx.set_cache_limit(int(cache_limit_gb * 2**30))
        self.model, self.tok = load(str(base))
        mx.eval(self.model.parameters())
        base_bytes = mx.get_active_memory()
        modules, lora_bytes = attach_lora(self.model, Path(info["adapter"]), info["rank"])
        self.vocab_size, hidden = self.model.model.embed_tokens.weight.shape
        self.heads = Heads(root, hidden)
        if any(i >= self.vocab_size for i in self.heads.column):
            raise ValueError("head token ids must be within the model's vocabulary")
        config = base / "generation_config.json"
        generation = json.loads(config.read_text()) if config.is_file() else {}
        self.generation_defaults = {
            k: generation[k] for k in ("temperature", "top_p", "top_k", "min_p") if generation.get(k) is not None
        }
        self.lock = threading.Lock()  # also guards Switch
        gib = 2**30
        self.summary = (
            f"base weights {base_bytes / gib:.2f} GiB; LoRA {modules} modules, {lora_bytes / gib:.3f} GiB (bfloat16); "
            f"heads {self.heads.weight.nbytes / 2**20:.1f} MiB; active {mx.get_active_memory() / gib:.2f} GiB"
        )

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text))

    def chat_ids(self, messages, kwargs) -> list[int]:
        return list(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs))

    def decision(self, prompt: list[int], ids: list[int], adapter: bool) -> list[float]:
        """Head logprobs: log-softmax over the heads, every other vocabulary entry -inf (as the vLLM plugin)."""
        with self.lock:
            Switch.on = adapter
            try:
                hidden = self.model.model(mx.array(prompt)[None])[0, -1]
            finally:
                Switch.on = False
            logits = self.heads.weight @ hidden.astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits)
            mx.eval(logprobs)
        return head_logprobs(logprobs.tolist(), self.heads.column, ids)

    def vocab(self, prompt: list[int], ids: list[int]) -> list[float]:
        """The base model's logprobs over its full vocabulary (bfloat16 logits upcast to float32)."""
        with self.lock:
            logits = self.model(mx.array(prompt)[None])[0, -1].astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits)
            return logprobs[mx.array(ids)].tolist()

    def generate(self, prompt_ids: list[int], params: dict):
        """Base-model generation (LoRA off). The lock is taken per token, so decision reads interleave with a
        running generation; each step builds its graph with the adapter off, and its KV cache is its own."""
        if params["seed"] is not None:
            mx.random.seed(int(params["seed"]))
        sampler = make_sampler(
            temp=params["temperature"], top_p=params["top_p"], min_p=params["min_p"], top_k=params["top_k"]
        )
        stops = params["stop"]
        steps = stream_generate(self.model, self.tok, prompt_ids, max_tokens=params["max_tokens"], sampler=sampler)
        text = ""
        try:
            while True:
                with self.lock:
                    Switch.on = False
                    step = next(steps, None)
                if step is None:
                    return
                text += step.text
                hit = next((s for s in stops if s in text), None)
                if hit:
                    yield text[: text.index(hit)], step.generation_tokens, "stop", True
                    return
                yield text, step.generation_tokens, step.finish_reason or "length", step.finish_reason is not None
        finally:
            with self.lock:
                steps.close()
