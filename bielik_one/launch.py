"""Launch the vLLM backend and the typed decision API for a downloaded Bielik One model package.

    bielik-one backend  PACKAGE_DIR [--fp8] [--port N] [--dry-run]
    bielik-one backend  PACKAGE_DIR --engine mlx [--base BASE_DIR] [--port N] [--dry-run]
    bielik-one frontend PACKAGE_DIR [--no-calibration] [--port N] [--backend-url URL] [--dry-run]

PACKAGE_DIR is a snapshot of one of the Bielik One Hugging Face repositories (merged or -Adapter). The package
kind is detected from its files; each kind starts with the profile validated for that package (see its card).
Runtime files and weights listed in SHA256SUMS.json are verified before anything starts. The frontend applies the package's
calibration.json unless --no-calibration is given. `--engine mlx` starts the Apple-silicon backend of an adapter
package instead of vLLM (bielik_one.mlx_backend); the frontend is the same.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import shlex
from pathlib import Path, PurePosixPath
import sys
import tempfile

TOKENIZER_FILES = [
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "*.model",
    "vocab.json",
    "merges.txt",
    "chat_template*",
    "*.jinja",
]
FP8 = [
    "--quantization",
    "fp8_per_tensor",
    "--quantization-config",
    '{"ignore":["*lm_head*"]}',
]
QWEN_BASE = "Qwen/Qwen3.5-4B"
# Environment the launcher sets for both processes (the validated configuration); --dry-run prints it with the command.
LAUNCH_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# kind -> default ports (backend, frontend)
PORTS = {
    "llama-adapter": (8025, 8026),
    "llama-merged": (8025, 8026),
    "qwen-adapter": (8013, 8012),
    "qwen-merged": (8013, 8012),
}

# These root-level files do not affect inference. Keep them in release manifests
# for package auditing, but do not let documentation edits block model startup.
NON_RUNTIME_FILES = frozenset({".gitattributes", "LICENSE", "README.md", "requirements.txt"})


# A Hugging Face commit id: unlike a branch or tag name, it always names the same files.
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def read_manifest(root: Path) -> bytes:
    path = root / "SHA256SUMS.json"
    if not path.is_file():
        raise ValueError("SHA256SUMS.json is missing; download the complete repository")
    return path.read_bytes()


def check_manifest(root: Path, expected: str | None) -> tuple[str, bytes]:
    """The manifest's digest and bytes; with `expected` (obtained independently of the package), they must match.

    verify() checks the files against SHA256SUMS.json, which travels with them: a package republished together with
    a matching manifest passes it. A digest recorded from a trusted source closes that gap. Pass the returned bytes
    to verify(), so the manifest that was pinned is the one the files are checked against.
    """
    raw = read_manifest(root)
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and not hmac.compare_digest(digest, expected.lower()):
        raise ValueError(
            f"SHA256SUMS.json has SHA-256 {digest}, not the pinned {expected.lower()}; "
            "this is not the package you pinned"
        )
    return digest, raw


def snapshot_revision(root: Path) -> str | None:
    """The commit id of a Hugging Face cache snapshot (snapshots/<commit>/), or None for any other directory."""
    return root.name if root.parent.name == "snapshots" and FULL_REVISION.fullmatch(root.name) else None


def resolve_package(spec: str, revision: str | None = None) -> Path:
    """A local package directory, or a Hugging Face repo id downloaded (or reused) in the HF cache."""
    path = Path(spec).expanduser()
    if path.is_dir():
        if revision:
            raise ValueError("--revision applies to a Hugging Face repo id, not a local directory")
        return path.resolve()
    if spec.count("/") == 1 and not spec.startswith((".", "/", "~")) and not path.exists():
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(spec, revision=revision))
    raise ValueError(f"{spec}: not a package directory or a Hugging Face repo id")


def verify(root: Path, manifest: bytes | None = None) -> dict:
    """Verify runtime files and all weights; ignore only known root-level auxiliary files."""
    sums = json.loads(manifest if manifest is not None else read_manifest(root))
    for relative, expected in sums.items():
        # Checked by name, not by resolving: HF cache snapshots are symlinks into blobs/.
        pure = PurePosixPath(relative)
        if pure.is_absolute() or "\\" in relative or ".." in pure.parts:
            raise ValueError("Invalid checksum path")
        if pure.as_posix() in NON_RUNTIME_FILES:
            continue
        target = root / pure
        if not target.is_file():
            raise ValueError(
                f"Missing file listed in SHA256SUMS.json: {relative}; download the complete repository"
            )
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
    if (root / "chat_template.jinja").is_file() and "chat_template.jinja" not in sums:
        raise ValueError(
            "chat_template.jinja is not part of this package (not in SHA256SUMS.json); remove it"
        )
    if (root / "heads.json").is_file():
        heads = json.loads((root / "heads.json").read_text())
        for name, field in [
            ("heads.safetensors", "weights_sha256"),
            ("adapter/adapter_model.safetensors", "adapter_sha256"),
        ]:
            if field in heads and name in sums and sums[name] != heads[field]:
                raise ValueError(f"Manifest mismatch: {name}")
    return sums


def detect(root: Path) -> str:
    conversion = json.loads((root / "conversion.json").read_text())
    serving = (
        json.loads((root / "serving.json").read_text())
        if (root / "serving.json").is_file()
        else None
    )
    if serving is not None:
        return "llama-adapter" if "base_model" in serving else "llama-merged"
    if conversion.get("merged_model"):
        return "qwen-merged"
    if (root / "heads.json").is_file() and (
        root / "adapter_model.safetensors"
    ).is_file():
        return "qwen-adapter"
    raise ValueError("Unrecognized package layout")


def calibration_args(root: Path, disabled: bool) -> list[str]:
    if disabled:
        return ["--no-calibration"]
    path = root / "calibration.json"
    return ["--calibration", str(path)] if path.is_file() else []


def backend(
    root: Path, kind: str, host: str, port: int, fp8: bool, eager: bool = False
) -> tuple[list[str], dict]:
    py = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    # bfloat16 by default: the reported results and the calibration are bfloat16. FP8 is opt-in.
    # CUDA graphs stay on. Eager matched the published validation runs; this path
    # prefers throughput. The decision readout runs outside the compiled backbone.
    # --eager: no CUDA graphs (shorter start, slower chat generation; decisions unchanged).
    quant = (FP8 if fp8 else []) + (["--enforce-eager"] if eager else [])
    common = [
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-logprobs",
        "256",
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-prefix-caching",
    ]
    plugin_env = {
        "VLLM_PLUGINS": "systemone_heads",
        "SYSTEMONE_HEADS_MANIFEST": str(root / "heads.json"),
    }
    if kind == "llama-adapter":
        c = json.loads((root / "serving.json").read_text())
        heads = json.loads((root / "heads.json").read_text())
        if (
            heads["base_revision"] != c["base_revision"]
            or heads["transformer_rank"] != c["rank"]
        ):
            raise ValueError("Serving configuration does not match heads")
        argv = (
            py
            + [
                c["base_model"],
                "--revision",
                c["base_revision"],
                "--hf-overrides",
                '{"architectures":["SystemOneLlamaForCausalLM"]}',
                "--served-model-name",
                c["base_alias"],
                "--enable-lora",
                "--lora-modules",
                f"decision={root / 'adapter'}",
                "--max-lora-rank",
                str(c["rank"]),
                "--max-loras",
                "1",
                "--max-cpu-loras",
                "1",
                "--lora-target-modules",
                "down_proj",
                "gate_proj",
                "k_proj",
                "o_proj",
                "q_proj",
                "up_proj",
                "v_proj",
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "2048",
                "--gpu-memory-utilization",
                "0.90",
                "--kv-cache-memory-bytes",
                str(c["kv_cache_memory_bytes"]),
            ]
            + common
            + quant
        )
        if c.get(
            "base_row_types"
        ):  # routed package: decision heads also on flagged base-model rows
            if not heads.get("base_row_heads"):
                raise ValueError(
                    "serving.json base_row_types requires heads.json base_row_heads"
                )
            argv += [
                "--logits-processors",
                "systemone_vllm_heads.processor:SystemOneBaseHeadsProcessor",
            ]
        return argv, plugin_env
    if kind == "llama-merged":
        c = json.loads((root / "serving.json").read_text())
        cfg = json.loads((root / "config.json").read_text())
        if cfg["architectures"] != ["LlamaForCausalLM"] or cfg.get(
            "tie_word_embeddings"
        ):
            raise ValueError("Expected native untied Llama")
        argv = (
            py
            + [
                str(root),
                "--served-model-name",  # chat alias first: vLLM's default for requests without a model
                c["general_alias"],
                "decision",
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "2048",
                "--gpu-memory-utilization",
                "0.90",
                "--kv-cache-memory-bytes",
                str(c["kv_cache_memory_bytes"]),
            ]
            + common
            + quant
        )
        return argv, {"VLLM_PLUGINS": "", "VLLM_MAX_NUM_BAD_WORDS": "2048"}
    if kind == "qwen-adapter":
        heads = json.loads((root / "heads.json").read_text())
        argv = (
            py
            + [
                QWEN_BASE,
                "--revision",
                heads["base_revision"],
                "--hf-overrides",
                '{"architectures":["SystemOneQwen3_5ForConditionalGeneration"]}',
                "--served-model-name",
                "qwen-base",
                "--enable-lora",
                "--lora-modules",
                f"decision={root}",
                "--max-lora-rank",
                "128",
                "--max-loras",
                "1",
                "--max-cpu-loras",
                "1",
                "--lora-target-modules",
                "down_proj",
                "gate_proj",
                "k_proj",
                "o_proj",
                "out_proj",
                "q_proj",
                "up_proj",
                "v_proj",
                "--max-num-seqs",
                "32",
                "--max-num-batched-tokens",
                "4096",
                "--kv-cache-memory-bytes",
                "4294967296",
                "--gpu-memory-utilization",
                "0.75",
                "--limit-mm-per-prompt",
                '{"image":0,"video":0}',
            ]
            + common
            + quant
        )
        return argv, plugin_env
    # qwen-merged: stored BF16
    argv = (
        py
        + [
            str(root),
            "--served-model-name",  # chat alias first: vLLM's default for requests without a model
            "qwen-general",
            "decision",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "2048",
            "--kv-cache-memory-bytes",
            "1610612736",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
        ]
        + common
        + quant
    )
    return argv, {"VLLM_PLUGINS": "", "VLLM_MAX_NUM_BAD_WORDS": "2048"}


# Chat parsers live in the upstream bielik-tools checkout, independently of
# the decision tokenizer/template pinned by the package.
BIELIK_CHAT_MODELS = {
    "speakleash/Bielik-11B-v3.1-Instruct": False,
    "speakleash/Bielik-11B-v3.0-Instruct": True,
    "speakleash/Bielik-Minitron-7B-v3.0-Instruct": True,
}


def bielik_chat_args(root: Path, kind: str, tools_dir: str | None = None) -> list[str]:
    """Required chat setup from one external bielik-tools checkout."""
    if kind == "native-tokens":
        # Original Bielik 3.2 already supplies its chat template.
        advanced_template = False
    else:
        if kind != "llama-adapter":
            raise ValueError("--bielik-tools-dir supports Bielik adapter or native packages only")
        model = json.loads((root / "serving.json").read_text())["base_model"]
        if model not in BIELIK_CHAT_MODELS:
            raise ValueError(f"--bielik-tools-dir: unsupported base model {model!r}")
        advanced_template = BIELIK_CHAT_MODELS[model]
    if not tools_dir:
        raise ValueError("Bielik tool and reasoning parsers are required: set BIELIK_TOOLS_DIR or --bielik-tools-dir to a trusted bielik-tools checkout")
    tools = Path(tools_dir).expanduser().resolve() / "tools"
    tool_parser = tools / "bielik_vllm_tool_parser.py"
    reasoning_parser = tools / "bielik_vllm_reasoning_parser.py"
    template = tools / "bielik_advanced_chat_template.jinja"
    required = [tool_parser, reasoning_parser] + ([template] if advanced_template else [])
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing bielik-tools file: {path}; use a checkout supporting vLLM 0.32")
    argv = [
        "--enable-auto-tool-choice",
        "--tool-parser-plugin", str(tool_parser),
        "--tool-call-parser", "bielik",
        "--reasoning-parser-plugin", str(reasoning_parser),
        "--reasoning-parser", "bielik",
        "--exclude-tools-when-tool-choice-none",
    ]
    # Bielik 3.1 already ships its tool/reasoning template. The 3.0 models
    # need the upstream advanced template for ordinary chat, not for decisions.
    if advanced_template:
        argv += ["--chat-template", str(template)]
    return argv


def chat_tools_dir(args):
    if args.no_bielik_chat_parsers:
        raise ValueError("Chat tool and reasoning parsers are required; --no-bielik-chat-parsers is no longer supported")
    return args.bielik_tools_dir if args.bielik_tools_dir is not None else os.environ.get("BIELIK_TOOLS_DIR")


def required_chat_args(root: Path, kind: str, args) -> list[str]:
    tools_dir = chat_tools_dir(args)
    if kind in ("llama-adapter", "native-tokens"):
        return bielik_chat_args(root, kind, tools_dir)
    if args.bielik_tools_dir is not None:
        return bielik_chat_args(root, kind, tools_dir)  # Reject explicit Bielik overrides on other model types.
    if kind == "qwen-adapter":
        return ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--reasoning-parser", "qwen3", "--exclude-tools-when-tool-choice-none"]
    return []


def frontend(
    root: Path,
    kind: str,
    host: str,
    port: int,
    backend_url: str,
    no_calibration: bool,
    dry_run: bool,
) -> list[str]:
    conversion = root / "conversion.json"
    if kind in ("llama-adapter", "qwen-adapter"):
        meta = json.loads(conversion.read_text())
        if kind == "llama-adapter":
            serving = json.loads((root / "serving.json").read_text())
            repo, rev, alias = (
                serving["base_model"],
                serving["base_revision"],
                serving["base_alias"],
            )
            if serving.get("base_row_types"):
                meta.update(base_row_types=serving["base_row_types"], base_alias=alias)
        else:
            repo, rev, alias = QWEN_BASE, meta["base_revision"], "qwen-base"
        if not dry_run:
            from huggingface_hub import snapshot_download

            meta["base_snapshot"] = snapshot_download(
                repo, revision=rev, allow_patterns=TOKENIZER_FILES
            )
        meta.update(
            adapter=str(root / "adapter"), heads_manifest=str(root / "heads.json")
        )
        if (
            root / "chat_template.jinja"
        ).is_file():  # pinned training template; verify() requires it in SHA256SUMS
            meta["chat_template_path"] = str(root / "chat_template.jinja")
        metadata = Path(tempfile.mkdtemp(prefix="bielik-one-")) / "conversion.json"
        metadata.write_text(json.dumps(meta))
        module, workers = (
            "bielik_one.api.remote_vllm",
            "8" if kind == "llama-adapter" else "32",
        )
    else:
        metadata, module, alias = (
            conversion,
            "bielik_one.api.merged_vllm",
            json.loads((root / "serving.json").read_text())["general_alias"]
            if kind == "llama-merged"
            else "qwen-general",
        )
        workers = "8" if kind == "llama-merged" else "32"
    return [
        sys.executable,
        "-m",
        module,
        "--metadata",
        str(metadata),
        "--backend-url",
        backend_url,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        workers,
        "--base-model",
        alias,
    ] + calibration_args(root, no_calibration)


def mlx_backend(
    root: Path,
    kind: str,
    host: str,
    port: int,
    base: str | None,
    cache_limit_gb: float | None,
    skip_base_check: bool,
) -> tuple[list[str], dict]:
    """The Apple-silicon backend (bielik_one.mlx_backend): adapter packages only, base model loaded once."""
    if kind not in ("llama-adapter", "qwen-adapter"):
        raise ValueError(
            "the MLX backend serves adapter packages (*-Adapter); merged packages need vLLM"
        )
    argv = [sys.executable, "-m", "bielik_one.mlx_backend", str(root), "--host", host, "--port", str(port)]
    if base is not None:
        argv += ["--base", str(Path(base).expanduser().resolve())]
    if cache_limit_gb is not None:
        argv += ["--cache-limit-gb", str(cache_limit_gb)]
    if skip_base_check:
        argv.append("--skip-base-check")
    return argv, {"TOKENIZERS_PARALLELISM": "false"}


def override(argv: list[str], flag: str, value) -> list[str]:
    """Replace the value of `flag` in a built command (profile defaults stay when `value` is None)."""
    if value is None:
        return argv
    if flag not in argv:
        raise ValueError(f"{flag} is not part of this package's backend profile")
    out = list(argv)
    out[out.index(flag) + 1] = str(value)
    return out


LOOPBACK = ("127.0.0.1", "localhost", "::1")
WILDCARD = ("0.0.0.0", "::", "")


def connect_host(bind: str) -> str:
    """An address a local client can use to reach a server bound to `bind` (URL form, IPv6 bracketed)."""
    host = "127.0.0.1" if bind in WILDCARD else bind
    return f"[{host}]" if ":" in host else host


def native_commands(root, component, args):
    """Explicit untouched-Llama profile; never auto-detect this for decision packages."""
    if args.engine != "vllm":
        raise ValueError("--native-tokens currently requires --engine vllm")
    cfg = json.loads((root / "config.json").read_text())
    if cfg.get("model_type") != "llama" or cfg.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("native tokens require an original LlamaForCausalLM checkpoint")
    if any((root / name).exists() for name in ("conversion.json", "heads.json", "adapter_config.json")):
        raise ValueError("native tokens require original weights, not a decision package or adapter")
    if component == "backend":
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(root),
                   "--served-model-name", "base", "--host", args.backend_host,
                   "--port", str(args.port or 8025), "--dtype", "bfloat16",
                   "--max-model-len", "32768", "--max-logprobs", "128",
                   "--logprobs-mode", "raw_logprobs", "--enable-prefix-caching",
                   "--max-num-seqs", str(args.max_num_seqs or 8),
                   "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.90"]
        command += required_chat_args(root, "native-tokens", args)
        if args.fp8:
            command += FP8
        if args.kv_cache_gb is not None:
            command += ["--kv-cache-memory-bytes", str(int(args.kv_cache_gb * 2**30))]
        if args.eager:
            command += ["--enforce-eager"]
        return command, {"VLLM_PLUGINS": ""}
    command = [sys.executable, "-m", "bielik_one.api.base_tokens", "--base-model", str(root),
               "--backend-url", args.backend_url or f"http://{connect_host(args.backend_host)}:8025",
               "--backend-model", "base", "--host", args.host, "--port", str(args.port or 8026),
               "--workers", str(args.max_num_seqs or 8), "--decision-prefix", args.decision_prefix]
    for flag, value in [("--queue-size", args.read_queue_size), ("--max-requests", args.max_requests),
                        ("--backend-timeout", args.backend_timeout), ("--request-timeout", args.request_timeout)]:
        if value is not None:
            command += [flag, str(value)]
    if args.no_calibration:
        command += ["--no-calibration"]
    if args.native_calibration is not None:
        command += ["--calibration", str(args.native_calibration.resolve())]
    if args.max_body_mb is not None:
        command += ["--max-body-mb", str(args.max_body_mb)]
    if args.error_format:
        command += ["--error-format", args.error_format]
    return command, {}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="bielik-one",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("component", choices=["backend", "frontend"])
    p.add_argument(
        "package",
        help="model package: a Hugging Face repo id (downloaded to the HF cache) or a local directory",
    )
    p.add_argument("--native-calibration", type=Path, help="native frontend: explicit calibration JSON (default: beside model)")
    p.add_argument("--read-queue-size", type=int, help="native frontend: queued reads beyond the worker pool (default: 64)")
    p.add_argument("--max-requests", type=int, help="native frontend: active processing requests (default: 32)")
    p.add_argument("--backend-timeout", type=float, help="native frontend: backend read timeout in seconds (default: 180)")
    p.add_argument("--request-timeout", type=float, help="native frontend: processing deadline in seconds (default: 300)")
    p.add_argument("--native-tokens", action="store_true", help="untouched Bielik 3.2: read original answer tokens; no adapter or heads")
    p.add_argument("--decision-prefix", choices=["empty-think", "plain"], default="empty-think", help="native tokens only: assistant suffix (default: documented empty-think contract)")
    p.add_argument("--revision", help="revision of a Hugging Face repo id (default: main)")
    chat = p.add_mutually_exclusive_group()
    chat.add_argument(
        "--bielik-tools-dir",
        metavar="DIR",
        help="vLLM backend: required bielik-tools checkout for Bielik tool/reasoning parsers "
        "(default: BIELIK_TOOLS_DIR); no parser files are copied or downloaded",
    )
    chat.add_argument(
        "--no-bielik-chat-parsers",
        action="store_true",
        help="removed: tool and reasoning parsers are required for adapter/native vLLM serving",
    )
    p.add_argument(
        "--manifest-sha256",
        metavar="HEX",
        help="refuse the package unless its SHA256SUMS.json has this SHA-256 (record it once from a trusted source; "
        "--dry-run prints it)",
    )
    p.add_argument(
        "--require-pinned-revision",
        action="store_true",
        help="refuse a Hugging Face repo id without --revision set to a full 40-character commit id, and a local "
        "directory without --manifest-sha256",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="frontend bind address (the typed API and the chat proxy)",
    )
    p.add_argument(
        "--backend-host",
        default="127.0.0.1",
        help="bind address of the raw vLLM backend; keep it on loopback unless the network is trusted",
    )
    p.add_argument("--port", type=int)
    p.add_argument("--backend-url")
    p.add_argument(
        "--fp8",
        action="store_true",
        help="backend with on-load FP8 weight quantization (less GPU memory; changes some decisions, see README)",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        metavar="N",
        help="backend: concurrent sequences (default from the validated profile); "
        "frontend: read pool at least N. Pass the same value to both.",
    )
    p.add_argument(
        "--kv-cache-gb",
        type=float,
        metavar="GB",
        help="backend KV cache size (default from the validated profile); raise it with --max-num-seqs",
    )
    p.add_argument(
        "--eager",
        action="store_true",
        help="backend without CUDA graphs (shorter start, slower chat generation; decisions are unchanged)",
    )
    p.add_argument(
        "--engine",
        choices=["vllm", "mlx"],
        default="vllm",
        help="backend engine: vllm (NVIDIA GPU, default) or mlx (Apple silicon, adapter packages; "
        "pip install 'bielik-one[mlx]')",
    )
    p.add_argument(
        "--base",
        metavar="DIR",
        help="--engine mlx: local base model at the revision the package pins (default: download it to the HF cache)",
    )
    p.add_argument(
        "--cache-limit-gb",
        type=float,
        metavar="GB",
        help="--engine mlx: MLX buffer cache limit (default: 4)",
    )
    p.add_argument(
        "--skip-base-check",
        action="store_true",
        help="--engine mlx: do not check the base model against the Hub's checksums of the pinned revision",
    )
    p.add_argument(
        "--max-body-mb",
        type=float,
        metavar="MB",
        help="frontend: largest request body in MiB, larger ones get HTTP 413 (default: 16)",
    )
    p.add_argument(
        "--error-format",
        choices=["default", "openai"],
        help='frontend: error bodies {"error", "code"} (default) or in the OpenAI format',
    )
    p.add_argument(
        "--no-calibration",
        action="store_true",
        help="frontend returns raw (temperature 1) probabilities",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="verify and print the command and its environment as JSON without starting it",
    )
    p.add_argument(
        "--shell",
        action="store_true",
        help="backend only: verify and print the vLLM command as one shell line (env VAR=... command ...) "
        "without starting it, to run vLLM yourself; see README, 'Running vLLM yourself'",
    )
    a = p.parse_args(argv)
    if (a.bielik_tools_dir is not None or a.no_bielik_chat_parsers) and (
        a.component != "backend" or a.engine != "vllm"
    ):
        p.error("--bielik-tools-dir and --no-bielik-chat-parsers are vLLM backend options")
    if a.shell and a.component != "backend":
        p.error(
            "--shell prints the vLLM backend command; start the API with "
            "'bielik-one frontend <package> --backend-url http://<host>:<port>'"
        )
    def given(pairs):
        return [flag for flag, value in pairs if value not in (None, False)]

    mlx_only = given(
        [("--base", a.base), ("--cache-limit-gb", a.cache_limit_gb), ("--skip-base-check", a.skip_base_check)]
    )
    if mlx_only and (a.engine != "mlx" or a.component != "backend"):
        p.error(f"{', '.join(mlx_only)}: options of 'bielik-one backend <package> --engine mlx'")
    if a.engine == "mlx" and a.component == "backend":
        vllm_only = given(
            [("--fp8", a.fp8), ("--eager", a.eager), ("--max-num-seqs", a.max_num_seqs), ("--kv-cache-gb", a.kv_cache_gb)]
        )
        if vllm_only:
            p.error(f"{', '.join(vllm_only)}: vLLM options, not used by --engine mlx")
        if a.cache_limit_gb is not None and not (math.isfinite(a.cache_limit_gb) and a.cache_limit_gb > 0):
            p.error("--cache-limit-gb must be positive")
    if a.max_num_seqs is not None and not 1 <= a.max_num_seqs <= 256:
        p.error("--max-num-seqs must be between 1 and 256")
    if a.kv_cache_gb is not None and not a.kv_cache_gb > 0:
        p.error("--kv-cache-gb must be positive")
    if a.max_body_mb is not None and not (math.isfinite(a.max_body_mb) and a.max_body_mb * 2**20 >= 1):
        p.error("--max-body-mb must be a finite size of at least one byte")
    if a.manifest_sha256 is not None and not SHA256_HEX.fullmatch(a.manifest_sha256.lower()):
        p.error("--manifest-sha256 must be 64 hexadecimal characters")
    if a.require_pinned_revision:
        # Checked before anything is downloaded.
        if Path(a.package).expanduser().is_dir():
            if a.native_tokens:
                p.error("--require-pinned-revision: a local native checkpoint has no revision or manifest to pin")
            if a.manifest_sha256 is None:
                p.error("--require-pinned-revision: a local directory has no revision; pass --manifest-sha256")
        elif a.revision is None or not FULL_REVISION.fullmatch(a.revision):
            p.error(
                "--require-pinned-revision: pass --revision with the full 40-character commit id "
                f"(got {a.revision or 'none, which means main'}); a branch, tag or short id can change"
            )
    if a.manifest_sha256 is not None and a.native_tokens:
        p.error("--manifest-sha256 does not apply to --native-tokens: original checkpoints have no SHA256SUMS.json")
    if a.native_calibration is not None and (not a.native_tokens or a.component != "frontend"):
        p.error("--native-calibration applies only to frontend --native-tokens")
    if a.native_calibration is not None and a.no_calibration:
        p.error("--native-calibration conflicts with --no-calibration")
    limits = (a.read_queue_size, a.max_requests, a.backend_timeout, a.request_timeout)
    if any(v is not None for v in limits) and (not a.native_tokens or a.component != "frontend"):
        p.error("native request limits apply only to 'frontend --native-tokens'")
    if a.read_queue_size is not None and not 0 <= a.read_queue_size <= 4096:
        p.error("--read-queue-size must be between 0 and 4096")
    if a.max_requests is not None and not 1 <= a.max_requests <= 256:
        p.error("--max-requests must be between 1 and 256")
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in (a.backend_timeout,a.request_timeout)):
        p.error("native timeouts must be finite and positive")
    try:
        root = resolve_package(a.package, a.revision)
    except ValueError as exc:
        p.error(str(exc))
    if a.native_tokens:
        if a.component == "backend" and a.host not in LOOPBACK:
            p.error("--host sets the frontend address; use --backend-host to bind the raw vLLM backend")
        try:
            command, extra = native_commands(root, a.component, a)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
        launch_env = {**LAUNCH_ENV, **extra}
        if a.shell:
            print(shlex.join(["env", *(f"{k}={v}" for k,v in launch_env.items()), *command]))
        elif a.dry_run:
            print(json.dumps({"kind":"native-tokens", "command":command, "env":launch_env, "checksums_verified":False}, indent=2))
        else:
            os.execvpe(command[0], command, {**os.environ, **launch_env})
        return 0
    if a.decision_prefix != "empty-think":
        p.error("--decision-prefix applies only to --native-tokens")
    try:
        digest, manifest = check_manifest(root, a.manifest_sha256)
        verify(root, manifest)
        kind = detect(root)
    except (ValueError, FileNotFoundError) as exc:
        p.error(f"{root}: {exc}")
    back_port, front_port = PORTS[kind]
    if a.component == "backend":
        if a.host not in LOOPBACK:
            p.error(
                "--host sets the frontend address; use --backend-host to bind the raw vLLM backend"
            )
    mlx = a.component == "backend" and a.engine == "mlx"
    if mlx:
        try:
            argv, extra = mlx_backend(
                root, kind, a.backend_host, a.port or back_port, a.base, a.cache_limit_gb, a.skip_base_check
            )
        except ValueError as exc:
            p.error(f"{root}: {exc}")
    elif a.component == "backend":
        argv, extra = backend(
            root, kind, a.backend_host, a.port or back_port, a.fp8, a.eager
        )
        try:
            argv += required_chat_args(root, kind, a)
        except ValueError as exc:
            p.error(str(exc))
        argv = override(argv, "--max-num-seqs", a.max_num_seqs)
        argv = override(
            argv,
            "--kv-cache-memory-bytes",
            None if a.kv_cache_gb is None else int(a.kv_cache_gb * 2**30),
        )
    else:
        url = a.backend_url or f"http://{connect_host(a.backend_host)}:{back_port}"
        argv, extra = (
            frontend(
                root,
                kind,
                a.host,
                a.port or front_port,
                url,
                a.no_calibration,
                a.dry_run or a.shell,
            ),
            {},
        )
        if a.max_body_mb is not None:
            argv += ["--max-body-mb", str(a.max_body_mb)]
        if a.error_format is not None:
            argv += ["--error-format", a.error_format]
        if a.max_num_seqs:
            workers = argv[argv.index("--workers") + 1]
            argv = override(argv, "--workers", max(int(workers), a.max_num_seqs))
    # The vLLM settings in LAUNCH_ENV do not apply to the MLX backend.
    launch_env = extra if mlx else {**LAUNCH_ENV, **extra}
    if a.shell:
        print(shlex.join(["env", *(f"{k}={v}" for k, v in launch_env.items()), *argv]))
        return 0
    env = os.environ.copy()
    env.update(launch_env)
    if a.dry_run:
        print(
            json.dumps(
                {
                    "kind": kind,
                    "command": argv,
                    "env": launch_env,
                    "checksums_verified": True,
                    "revision": snapshot_revision(root),
                    "manifest_sha256": digest,
                    "manifest_pinned": a.manifest_sha256 is not None,
                },
                indent=2,
            )
        )
        return 0
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    sys.exit(main())
