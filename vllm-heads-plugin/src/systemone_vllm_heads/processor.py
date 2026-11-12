"""Logits processor for decision heads on base-model rows (vLLM 0.32.0 public logits-processor interface).

Requests that carry vllm_xargs {"systemone_head": "base"} and no LoRA adapter get decision-head outputs instead of
vocabulary logits; every other row is returned unchanged, so ordinary generation of the base model is unaffected.
Enable with --logits-processors systemone_vllm_heads.processor:SystemOneBaseHeadsProcessor and a heads manifest
with "base_row_heads": true.
"""
from vllm.v1.sample.logits_processor import LogitsProcessor

from .core import BASE_HEAD_FLAG, BASE_HEAD_VALUE, LAST_STEP, apply_base_heads, update_flags


class SystemOneBaseHeadsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        extra = sampling_params.extra_args or {}
        if BASE_HEAD_FLAG in extra and extra[BASE_HEAD_FLAG] != BASE_HEAD_VALUE:
            raise ValueError(f"{BASE_HEAD_FLAG} must be {BASE_HEAD_VALUE!r}")

    def __init__(self, vllm_config, device, is_pin_memory):
        import json, os
        from pathlib import Path
        path = os.environ.get("SYSTEMONE_HEADS_MANIFEST")
        if not path or not json.loads(Path(path).read_text()).get("base_row_heads"):
            raise ValueError("SystemOneBaseHeadsProcessor requires a heads manifest with base_row_heads enabled")
        self.flags = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        update_flags(self.flags, batch_update)

    def apply(self, logits):
        return apply_base_heads(logits, sorted(self.flags), LAST_STEP)
