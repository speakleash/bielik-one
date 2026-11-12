"""Temperature scaling of answer distributions."""

from __future__ import annotations


import torch


def at_temperature(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """A single temperature does not change the argmax. It only moves the probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if abs(temperature - 1.0) < 1e-8:
        return probs
    logp = probs.double().clamp_min(1e-12).log()
    # Center before scaling: even a subnormal T leaves a zero maximum.
    # Other entries may safely saturate at -inf instead of making every
    # entry -inf (whose softmax would be NaN). Float64 preserves tiny T.
    scaled = (logp - logp.amax(dim=-1, keepdim=True)) / temperature
    return torch.softmax(scaled, dim=-1).to(probs.dtype)
