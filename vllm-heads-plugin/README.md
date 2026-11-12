# systemone-vllm-heads

vLLM 0.32.0 plugin that adds the Bielik One decision heads to Qwen3.5 and Bielik (Llama-architecture) models served with a LoRA adapter. It is installed and started by `bielik-one`; see the main README.

- Registers `SystemOneQwen3_5ForConditionalGeneration` and `SystemOneLlamaForCausalLM` (opt-in through `--hf-overrides`); the standard model classes are untouched.
- Rows of requests that use the decision adapter get the decision-head outputs (choice, binary, score, labels and an anchor) instead of the vocabulary projection; all other rows, including ordinary generation through the base model, keep the original full projection.
- For routed packages (heads manifest v2, `base_row_heads: true`), `systemone_vllm_heads.processor:SystemOneBaseHeadsProcessor` applies the decision heads to base-model rows that the frontend flags with `vllm_xargs={"systemone_head": "base"}`, so those questions need no LoRA.
- Heads are loaded on CPU from `SYSTEMONE_HEADS_MANIFEST` (checksummed) and moved to the language-model head's device. One fixed adapter, TP=PP=1, V1 model runner.

Install into the same environment as vLLM, without touching its dependencies:

```bash
pip install --no-deps "systemone-vllm-heads @ git+https://github.com/speakleash/bielik-one#subdirectory=vllm-heads-plugin"
```

`bielik-one backend <package>` sets `VLLM_PLUGINS=systemone_heads`, `SYSTEMONE_HEADS_MANIFEST` and, for routed packages, `--logits-processors`.
