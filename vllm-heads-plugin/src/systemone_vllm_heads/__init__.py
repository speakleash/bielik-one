"""Opt-in architecture registration; existing Qwen and Llama architectures are untouched."""


def register():
    from importlib.metadata import version
    import os
    from vllm import ModelRegistry

    if version("vllm") != "0.32.0":
        if os.environ.get("SYSTEMONE_HEADS_MANIFEST"):
            raise RuntimeError("SystemOne heads currently require vLLM 0.32.0")
        return
    ModelRegistry.register_model(
        "SystemOneQwen3_5ForConditionalGeneration",
        "systemone_vllm_heads.model:SystemOneQwen3_5ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "SystemOneLlamaForCausalLM",
        "systemone_vllm_heads.llama_model:SystemOneLlamaForCausalLM",
    )
