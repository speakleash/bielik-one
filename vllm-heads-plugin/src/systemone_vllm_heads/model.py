"""vLLM 0.32.0 integration; one fixed adapter, TP=PP=1, native base unchanged."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm import envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from .core import DecisionReadout, package_file

TARGETS = {"down_proj", "gate_proj", "k_proj", "o_proj", "out_proj", "q_proj", "up_proj", "v_proj"}


class SystemOneQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix="model"):
        parallel = vllm_config.parallel_config
        lora = vllm_config.lora_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("SystemOne heads currently require TP=PP=1")
        if (lora is None or lora.max_lora_rank != 128 or lora.max_loras != 1
                or lora.max_cpu_loras != 1 or set(lora.target_modules or []) != TARGETS):
            raise ValueError("Require rank128, max_loras=max_cpu_loras=1 and exact transformer targets")
        if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
            raise ValueError("Dynamic adapter replacement is unsupported: heads belong to one fixed adapter")
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("Use VLLM_USE_V2_MODEL_RUNNER=0 with this initial plugin")
        path = os.environ.get("SYSTEMONE_HEADS_MANIFEST")
        if not path:
            raise ValueError("SYSTEMONE_HEADS_MANIFEST is required")
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") not in ("systemone-native-heads-v1", "systemone-native-heads-v2"):
            raise ValueError("Unsupported heads manifest")
        if (manifest["format"] == "systemone-native-heads-v2") != bool(manifest.get("base_row_heads")):
            raise ValueError("Heads manifest v2 is exactly the routed base-row-heads layout")
        if vllm_config.model_config.revision != manifest["base_revision"]:
            raise ValueError("Pin the base revision recorded in the heads manifest")
        weight_path = package_file(manifest_path.parent, manifest["weights_file"])
        if hashlib.file_digest(weight_path.open("rb"), "sha256").hexdigest() != manifest["weights_sha256"]:
            raise ValueError("Head weight checksum mismatch")
        weights = load_file(str(weight_path), device="cpu")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if weights["weight"].shape[1] != self.config.text_config.hidden_size:
            raise ValueError("Head hidden size does not match the base model")
        native_processor = self.language_model.logits_processor
        if native_processor.scale != 1 or native_processor.soft_cap is not None:
            raise ValueError("Only unscaled Qwen logits without soft-cap are supported")
        readout = DecisionReadout(weights["weight"], weights["token_ids"],
                                  vocab_size=self.config.text_config.vocab_size,
                                  anchor_id=manifest["anchor_id"],
                                  base_row_heads=bool(manifest.get("base_row_heads", False)))
        self.systemone_readout = readout.to(self.language_model.lm_head.weight.device)
        self._systemone_punica = None

    def _sampler_indices(self):
        if self._systemone_punica is None:
            for module in self.modules():
                if isinstance(module, BaseLayerWithLoRA):
                    self._systemone_punica = module.punica_wrapper
                    break
            if self._systemone_punica is None:
                raise RuntimeError("Decision readout requires native transformer LoRA metadata")
        return self._systemone_punica.sampler_indices

    def compute_logits(self, hidden_states):
        if hidden_states.shape[0] != self._sampler_indices().numel():
            # Prompt-logprob rows have no per-row LoRA mapping: ordinary vocabulary logits, no engine failure.
            return super().compute_logits(hidden_states)
        if os.environ.get("SYSTEMONE_SKIP_NATIVE_PROJECTION", "1") == "1":
            return self.systemone_readout.selective(
                hidden_states, self._sampler_indices(), super().compute_logits)
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        return self.systemone_readout(hidden_states, logits, self._sampler_indices())

    def compute_logits_local(self, hidden_states):
        if get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("Only TP=1 is supported")
        return self.compute_logits(hidden_states)
