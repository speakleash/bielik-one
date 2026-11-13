"""Apple-silicon (MLX) backend for adapter packages: one copy of the base model serves chat and decisions.

    bielik-one backend PACKAGE --engine mlx [--base BASE_DIR] [--port N]
    bielik-one frontend PACKAGE

The MLX counterpart of `bielik-one backend` for an adapter package (vLLM + LoRA + the systemone-vllm-heads plugin).
It answers the requests `bielik-one frontend` sends to vLLM, so the frontend runs on top of it unchanged:

  model "decision"     one-token reads with the LoRA on; the logits of the head token ids are the decision heads
                       (float32, heads.weight @ final hidden state), every other vocabulary entry is -inf
  model <base alias>   the unchanged base model (LoRA off): one-token reads of its vocabulary logprobs, and with
                       vllm_xargs {"systemone_head": "base"} the decision heads on a base-model row (routed packages)
  /v1/chat/completions, /v1/completions (text): generation by the unchanged base model, streaming or not

The LoRA is not merged: every targeted Linear keeps its base weight and gets the two low-rank matrices next to it,
switched on only while a decision read builds its graph. One forward pass runs at a time (Apple GPUs have one
queue); a chat generation takes the lock per token, so decision reads interleave with running chats.

The base model is the repository and revision pinned by the package (heads.json, serving.json). Without --base it is
downloaded to the Hugging Face cache at that revision; a local --base directory must hold the same files. Weights,
config and tokenizer files are checked against the Hub's checksums of the pinned revision before loading.

This module holds the HTTP protocol and the checks, testable without MLX; the model code is in mlx_model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from bielik_one import launch

# What vLLM reports for a -inf processed logprob; the frontend treats <= -9990 as "not a head".
CLAMP = -9999.0
MAX_MODEL_LEN = 32768
BASE_HEAD_FLAG, BASE_HEAD_VALUE = "systemone_head", "base"
# Base repository files the backend reads: mlx_lm loads model*.safetensors, the config and the tokenizer.
BASE_FILES = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model*.safetensors.index.json",
    *launch.TOKENIZER_FILES,
]


# -- package and base model -------------------------------------------------------------------------------------


def adapter_package(root: Path, kind: str) -> dict:
    """The base repository, revision and alias an adapter package pins, and where its LoRA lives."""
    if kind not in ("llama-adapter", "qwen-adapter"):
        raise ValueError(f"a {kind} package; the MLX backend serves adapter packages (*-Adapter)")
    heads = json.loads((root / "heads.json").read_text())
    if kind == "llama-adapter":
        serving = json.loads((root / "serving.json").read_text())
        if heads["base_revision"] != serving["base_revision"] or heads["transformer_rank"] != serving["rank"]:
            raise ValueError("Serving configuration does not match heads")
        return {
            "repo": serving["base_model"],
            "revision": serving["base_revision"],
            "alias": serving["base_alias"],
            "adapter": root / "adapter",
            "rank": serving["rank"],
            "base_row_heads": bool(serving.get("base_row_types")),
        }
    return {
        "repo": launch.QWEN_BASE,
        "revision": heads["base_revision"],
        "alias": "qwen-base",
        "adapter": root,
        "rank": heads.get("transformer_rank"),
        "base_row_heads": False,
    }


def check_adapter_manifest(root: Path, info: dict, sums: dict) -> None:
    """The LoRA and the heads are the ones heads.json was built for (launch.verify has checked SHA256SUMS.json)."""
    heads = json.loads((root / "heads.json").read_text())
    adapter = (Path(info["adapter"]) / "adapter_model.safetensors").relative_to(root).as_posix()
    for name, field in [(adapter, "adapter_sha256"), (heads.get("weights_file"), "weights_sha256")]:
        if sums.get(name) is None or sums[name] != heads.get(field):
            raise ValueError(f"Manifest mismatch: {name} is not the file heads.json names ({field})")


def git_blob_sha1(path: Path) -> str:
    """The Hub's id of a file stored without LFS (a git blob hash)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wanted(name: str) -> bool:
    return "/" not in name and any(fnmatch(name, pattern) for pattern in BASE_FILES)


def hub_checksums(repo: str, revision: str) -> dict[str, tuple[str, str]]:
    """{file: ("sha256" | "git", digest)} of the base files at the pinned revision, from the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if not info.sha or not info.sha.startswith(revision):
        raise ValueError(f"{repo}: the Hub resolved revision {revision} to {info.sha}")
    return checksums_from_siblings(info.siblings)


def checksums_from_siblings(siblings) -> dict[str, tuple[str, str]]:
    out = {}
    for sibling in siblings:
        if not wanted(sibling.rfilename):
            continue
        lfs = getattr(sibling, "lfs", None)
        out[sibling.rfilename] = ("sha256", lfs.sha256) if lfs else ("git", sibling.blob_id)
    if not any(fnmatch(name, "model*.safetensors") for name in out):
        raise ValueError("the pinned base revision lists no model*.safetensors")
    return out


def verify_base(base: Path, expected: dict[str, tuple[str, str]]) -> None:
    """The base directory holds exactly the pinned weights and the pinned config and tokenizer files."""
    for name, (algorithm, digest) in sorted(expected.items()):
        path = base / PurePosixPath(name)
        if not path.is_file():
            raise ValueError(f"base model: {name} of the pinned revision is missing")
        actual = sha256(path) if algorithm == "sha256" else git_blob_sha1(path)
        if actual != digest:
            raise ValueError(f"base model: {name} differs from the pinned revision")
    # mlx_lm loads every model*.safetensors in the directory: an extra one would change the weights.
    extra = sorted(
        p.name for p in base.glob("model*.safetensors") if p.name not in expected
    )
    if extra:
        raise ValueError(f"base model: {', '.join(extra)} not part of the pinned revision")


# -- adapter -----------------------------------------------------------------------------------------------------

LORA_KEY = re.compile(r"(layers\.\d+\..+)\.lora_([AB])\.weight$")
MODULE_SUFFIX = re.compile(r"(layers\.\d+\..+)$")


def lora_scale(config: dict, rank: int | None = None) -> float:
    """alpha / r of a plain PEFT LoRA; anything the backend would apply differently from PEFT is refused."""
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("adapter: only PEFT LoRA is supported")
    unsupported = [
        key
        for key in (
            "use_dora", "use_rslora", "rank_pattern", "alpha_pattern", "modules_to_save",
            "layer_replication", "trainable_token_indices", "fan_in_fan_out", "lora_bias",
        )
        if config.get(key)
    ]
    if config.get("bias", "none") != "none":
        unsupported.append("bias")
    if unsupported:
        raise ValueError(f"adapter: unsupported LoRA options {unsupported}")
    if rank is not None and config["r"] != rank:
        raise ValueError(f"adapter rank {config['r']} does not match the package manifest ({rank})")
    return config["lora_alpha"] / config["r"]


def lora_pairs(keys) -> dict[str, dict[str, str]]:
    """PEFT tensor names grouped by module suffix ("layers.N...."): {suffix: {"A": key, "B": key}}."""
    pairs: dict[str, dict[str, str]] = {}
    for key in keys:
        hit = LORA_KEY.search(key)
        if not hit:
            raise ValueError(f"unexpected adapter tensor {key}")
        pairs.setdefault(hit.group(1), {})[hit.group(2)] = key
    incomplete = [suffix for suffix, ab in pairs.items() if set(ab) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"adapter: lora_A/lora_B missing for {incomplete[:3]}")
    return pairs


def module_index(paths) -> dict[str, str]:
    """Model module paths by their "layers.N...." suffix, the part PEFT and MLX names share."""
    index, seen = {}, set()
    for path in paths:
        hit = MODULE_SUFFIX.search(path)
        if not hit:
            continue
        suffix = hit.group(1)
        if suffix in seen:
            index.pop(suffix, None)  # ambiguous: refused if the adapter targets it
            continue
        seen.add(suffix)
        index[suffix] = path
    return index


# -- decision reads ----------------------------------------------------------------------------------------------


def head_logprobs(values: list[float], column: dict[int, int], ids: list[int]) -> list[float]:
    """Requested logprobs from head-only logits: head ids from their row, every other id is -inf (clamped)."""
    return [values[column[i]] if i in column else CLAMP for i in ids]


def read_kind(body: dict, alias: str, base_row_heads: bool) -> str:
    """How a one-token read is answered: "decision" (LoRA + heads), "base_heads" (heads on a base-model row) or
    "vocab" (the base model's vocabulary). As in vLLM, the base-row flag is honoured only when the package routes
    question types to the base model (its logits processor is loaded only then)."""
    if body.get("model") == "decision":
        return "decision"
    flagged = (body.get("vllm_xargs") or {}).get(BASE_HEAD_FLAG) == BASE_HEAD_VALUE
    return "base_heads" if flagged and base_row_heads else "vocab"


def sampling(body: dict, defaults: dict, prompt_tokens: int) -> dict:
    """Generation parameters as vLLM reads them: request values, else the model's generation_config, else vLLM's
    defaults. Without max_tokens a generation may fill the context."""

    def pick(name, fallback):
        value = body.get(name)
        return defaults.get(name, fallback) if value is None else value

    top_k = int(pick("top_k", 0))
    top_p = float(pick("top_p", 1.0))
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    room = MAX_MODEL_LEN - prompt_tokens
    if max_tokens is None:
        max_tokens = room
    max_tokens = int(max_tokens)
    if max_tokens < 1 or max_tokens > room:
        raise ValueError(
            f"max_tokens must be between 1 and {room} ({prompt_tokens} prompt tokens, context {MAX_MODEL_LEN})"
        )
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ValueError("stop must be a string or a list of strings")
    if (body.get("n") or 1) != 1:
        raise ValueError("n > 1 is not supported")
    return {
        "temperature": float(pick("temperature", 1.0)),
        "top_p": top_p if 0 < top_p < 1 else 0.0,
        "top_k": max(top_k, 0),
        "min_p": float(pick("min_p", 0.0)),
        "max_tokens": max_tokens,
        "stop": [s for s in stop if s],
        "seed": body.get("seed"),
    }


def token_prompt(prompt) -> bool:
    return isinstance(prompt, list) and bool(prompt) and all(
        isinstance(t, int) and not isinstance(t, bool) for t in prompt
    )


def build_app(runner, alias: str, base_row_heads: bool = False):
    """The vLLM routes `bielik-one frontend` uses, over a runner (mlx_model.MlxRunner, or a fake in tests).

    A runner provides decision(prompt, ids, adapter), vocab(prompt, ids), encode(text), chat_ids(messages, kwargs),
    generation_defaults and generate(prompt_ids, params) yielding (text so far, tokens, finish reason, done).
    """
    app = FastAPI(title="bielik-one MLX backend")
    names = [alias, "decision"]

    def error(status, message):
        kind = "NotFoundError" if status == 404 else "BadRequestError"
        return JSONResponse(
            status_code=status,
            content={"error": {"message": message, "type": kind, "param": None, "code": status}},
        )

    async def json_body(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        # As vLLM lists them: the base model under its served name, the LoRA with its parent.
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "owned_by": "bielik-one", "root": alias, "parent": None},
                {"id": "decision", "object": "model", "owned_by": "bielik-one", "root": "decision", "parent": alias},
            ],
        }

    def generation(body, prompt_ids, chat: bool):
        model = body.get("model") or alias
        if model == "decision":
            return error(400, "text generation is served by the base model only")
        if model != alias:
            return error(404, f"The model `{model}` does not exist.")
        try:
            params = sampling(body, runner.generation_defaults, len(prompt_ids))
        except (ValueError, TypeError) as exc:
            return error(400, str(exc))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        kind = "chat.completion" if chat else "text_completion"
        if body.get("stream"):

            def events():
                sent = ""
                if chat:
                    first = {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                    yield sse(rid, kind + ".chunk", created, model, first)
                for text, _, finish, done in runner.generate(prompt_ids, params):
                    delta, sent = text[len(sent):], text
                    if chat:
                        choice = {"index": 0, "delta": {"content": delta}, "finish_reason": finish if done else None}
                    else:
                        choice = {"index": 0, "text": delta, "finish_reason": finish if done else None}
                    yield sse(rid, kind + ".chunk" if chat else kind, created, model, choice)
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        text, tokens, finish = "", 0, "length"
        for text, tokens, finish, _ in runner.generate(prompt_ids, params):
            pass
        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            if chat
            else {"index": 0, "text": text, "finish_reason": finish, "logprobs": None}
        )
        return {
            "id": rid,
            "object": kind,
            "created": created,
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": tokens,
                "total_tokens": len(prompt_ids) + tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await json_body(request)
        if body is None:
            return error(400, "expected a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return error(400, "messages must be a non-empty list")
        try:
            ids = runner.chat_ids(messages, dict(body.get("chat_template_kwargs") or {}))
        except Exception as exc:  # template errors: the request is the client's to change
            return error(400, f"chat template: {exc}")
        if len(ids) >= MAX_MODEL_LEN:
            return error(400, f"prompt of {len(ids)} tokens exceeds the context ({MAX_MODEL_LEN})")
        return await run_in_threadpool(generation, body, list(ids), True)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await json_body(request)
        if body is None:
            return error(400, "expected a JSON object")
        prompt, model = body.get("prompt"), body.get("model") or alias
        if model not in names:
            return error(404, f"The model `{model}` does not exist.")
        ids = body.get("logprob_token_ids")
        if model == "decision" or ids is not None:
            # A decision read, as bielik-one frontend sends it.
            if not token_prompt(prompt):
                return error(400, "decision reads take a tokenized prompt")
            if body.get("max_tokens", 1) != 1 or not token_prompt(ids):
                return error(400, "decision reads are one-token reads with logprob_token_ids")
            if len(prompt) >= MAX_MODEL_LEN:
                return error(400, f"prompt of {len(prompt)} tokens exceeds the context ({MAX_MODEL_LEN})")
            kind = read_kind(body, alias, base_row_heads)
            if kind == "vocab":
                values = await run_in_threadpool(runner.vocab, prompt, ids)
            else:
                values = await run_in_threadpool(runner.decision, prompt, ids, kind == "decision")
            return {
                "id": "cmpl-" + uuid.uuid4().hex,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": "",
                    "finish_reason": "length",
                    "logprobs": {"top_logprobs": [{f"token_id:{i}": v for i, v in zip(ids, values)}]},
                }],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1, "total_tokens": len(prompt) + 1},
            }
        if token_prompt(prompt):
            prompt_ids = list(prompt)
        elif isinstance(prompt, str) and prompt:
            prompt_ids = runner.encode(prompt)
        else:
            return error(400, "prompt must be a string or a list of token ids")
        if len(prompt_ids) >= MAX_MODEL_LEN:
            return error(400, f"prompt of {len(prompt_ids)} tokens exceeds the context ({MAX_MODEL_LEN})")
        return await run_in_threadpool(generation, body, prompt_ids, False)

    return app


def sse(rid, kind, created, model, choice) -> str:
    return "data: " + json.dumps(
        {"id": rid, "object": kind, "created": created, "model": model, "choices": [choice]}
    ) + "\n\n"


# -- entry point -------------------------------------------------------------------------------------------------


def resolve_base(info: dict, base: str | None, check: bool) -> Path:
    """The base model directory: --base, or the pinned revision downloaded to the HF cache; verified unless
    check is False."""
    if base is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(info["repo"], revision=info["revision"], allow_patterns=BASE_FILES))
    else:
        path = Path(base).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"--base {base}: not a directory")
    if check:
        print(f"verifying base model {info['repo']}@{info['revision'][:7]} in {path}", flush=True)
        verify_base(path, hub_checksums(info["repo"], info["revision"]))
    else:
        print(f"WARNING: base model in {path} not verified against {info['repo']}@{info['revision']}", flush=True)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bielik_one.mlx_backend",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("package", type=Path, help="adapter package directory (verified with its SHA256SUMS.json)")
    p.add_argument("--base", help="local directory of the base model at the pinned revision (default: download it)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--cache-limit-gb", type=float, default=4.0, help="MLX buffer cache limit (default: 4)")
    p.add_argument(
        "--skip-base-check",
        action="store_true",
        help="load the base model without checking it against the Hub's checksums of the pinned revision",
    )
    a = p.parse_args(argv)
    root = a.package.expanduser().resolve()
    try:
        sums = launch.verify(root)
        info = adapter_package(root, launch.detect(root))
        check_adapter_manifest(root, info, sums)
        base = resolve_base(info, a.base, not a.skip_base_check)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        p.error(f"{root}: {exc}")

    import uvicorn

    from bielik_one.mlx_model import MlxRunner

    runner = MlxRunner(base, root, info, cache_limit_gb=a.cache_limit_gb)
    print(runner.summary, flush=True)
    app = build_app(runner, info["alias"], info["base_row_heads"])
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
