"""Fixed-adapter Llama decision readouts, vLLM 0.32.0, TP=PP=1."""
import hashlib
import json
import os
from pathlib import Path

from safetensors.torch import load_file
from vllm import envs
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.models.llama import LlamaForCausalLM
from .core import DecisionReadout, package_file

TARGETS = {"down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"}

class SystemOneLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        lora = vllm_config.lora_config
        path = Path(os.environ["SYSTEMONE_HEADS_MANIFEST"])
        manifest = json.loads(path.read_text())
        if (manifest.get("format") not in ("systemone-native-heads-v1", "systemone-native-heads-v2")
                or manifest.get("architecture") != "LlamaForCausalLM"):
            raise ValueError("Expected Llama heads manifest")
        if (manifest["format"] == "systemone-native-heads-v2") != bool(manifest.get("base_row_heads")):
            raise ValueError("Heads manifest v2 is exactly the routed base-row-heads layout")
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Only TP=PP=1 supported")
        if (lora is None or lora.max_lora_rank != manifest["transformer_rank"] or lora.max_loras != 1
                or lora.max_cpu_loras != 1 or set(lora.target_modules or []) != TARGETS):
            raise ValueError("Require fixed adapter rank and exact Llama transformer targets")
        if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING or envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("Dynamic adapters and V2 runner unsupported")
        if vllm_config.model_config.revision != manifest["base_revision"]:
            raise ValueError("Pin manifest base revision")
        weights_path = package_file(path.parent, manifest["weights_file"])
        with weights_path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != manifest["weights_sha256"]:
                raise ValueError("Head checksum mismatch")
        weights = load_file(str(weights_path), device="cpu")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if (weights['weight'].shape[1] != self.config.hidden_size or
                self.logits_processor.scale != 1 or self.logits_processor.soft_cap is not None):
            raise ValueError("Unsupported hidden size or logits scaling")
        self.systemone_readout = DecisionReadout(weights['weight'], weights['token_ids'],
            vocab_size=self.config.vocab_size, anchor_id=manifest['anchor_id'],
            base_row_heads=bool(manifest.get('base_row_heads', False))).to(self.lm_head.weight.device)
        self._systemone_punica = None

    def _sampler_indices(self):
        if self._systemone_punica is None:
            for module in self.modules():
                if isinstance(module, BaseLayerWithLoRA):
                    self._systemone_punica = module.punica_wrapper
                    break
            if self._systemone_punica is None:
                raise RuntimeError("Missing LoRA sampler mapping")
        return self._systemone_punica.sampler_indices

    def compute_logits(self, hidden_states):
        sampler = self._sampler_indices()
        if hidden_states.shape[0] != sampler.numel():
            # Prompt-logprob computation passes prompt rows, not one row per request: no per-row LoRA mapping.
            # Return ordinary vocabulary logits instead of failing inside the engine.
            return super().compute_logits(hidden_states)
        return self.systemone_readout.selective(hidden_states, sampler, super().compute_logits)

    def compute_logits_local(self, hidden_states):
        return self.compute_logits(hidden_states)
