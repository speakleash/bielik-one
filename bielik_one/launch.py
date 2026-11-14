"""Launch the vLLM backend and the typed decision API for a downloaded Bielik One model package.

    bielik-one backend  PACKAGE_DIR [--fp8] [--port N] [--dry-run]
    bielik-one frontend PACKAGE_DIR [--no-calibration] [--port N] [--backend-url URL] [--dry-run]

PACKAGE_DIR is a snapshot of one of the Bielik One Hugging Face repositories (merged or -Adapter). The package
kind is detected from its files; each kind starts with the profile validated for that package (see its card).
Runtime files and weights listed in SHA256SUMS.json are verified before anything starts. The frontend applies the package's
calibration.json unless --no-calibration is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def _package_path(name) -> PurePosixPath:
    """A file named by a manifest: relative, inside the package. Checked by name, not by resolving: HF cache snapshots
    are symlinks into blobs/."""
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid checksum path")
    pure = PurePosixPath(name)
    if pure.is_absolute() or "\\" in name or ".." in pure.parts:
        raise ValueError("Invalid checksum path")
    return pure


def _shipped_files(root: Path) -> set[str]:
    """Every file in the package that must be listed: all of them except SHA256SUMS.json, the root-level auxiliary
    files, and hidden paths (Hugging Face metadata such as .cache/ from --local-dir or .eval_results/)."""
    shipped = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        name = relative.as_posix()
        if name != "SHA256SUMS.json" and name not in NON_RUNTIME_FILES:
            shipped.add(name)
    return shipped


def _required_files(root: Path, kind: str, heads: dict | None) -> set[str]:
    """Files the package kind cannot run without; each must have a manifest entry."""
    required = {"conversion.json"}
    if kind.startswith("llama"):
        required.add("serving.json")
    if kind.endswith("merged"):
        required |= {"config.json", "tokenizer.json", "tokenizer_config.json"}
        index = root / "model.safetensors.index.json"
        if index.is_file():
            required.add(index.name)
            weight_map = json.loads(index.read_text()).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("model.safetensors.index.json has no weight_map")
            required |= {_package_path(name).as_posix() for name in weight_map.values()}
        else:
            required.add("model.safetensors")
    else:
        adapter = "adapter/" if kind == "llama-adapter" else ""
        required |= {"heads.json", f"{adapter}adapter_config.json", f"{adapter}adapter_model.safetensors"}
        if heads is not None:
            weights_file = heads.get("weights_file")
            if not isinstance(weights_file, str):
                raise ValueError("heads.json must name its weights_file")
            required.add(_package_path(weights_file).as_posix())
    return required


def verify(root: Path) -> dict:
    """Verify the package against SHA256SUMS.json: every listed file must match, every file the package ships must be
    listed (except the root-level auxiliary files), and the files its kind needs to run must be among them."""
    path = root / "SHA256SUMS.json"
    if not path.is_file():
        raise ValueError("SHA256SUMS.json is missing; download the complete repository")
    sums = json.loads(path.read_text())
    if not isinstance(sums, dict) or not sums:
        raise ValueError("SHA256SUMS.json must be a non-empty object of file names and SHA-256 digests")
    for relative, expected in sums.items():
        pure = _package_path(relative)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid SHA-256 digest for {relative} in SHA256SUMS.json")
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
    unlisted = sorted(_shipped_files(root) - set(sums))
    if unlisted:
        raise ValueError(
            f"Not listed in SHA256SUMS.json: {', '.join(unlisted[:5])}"
            + (f" and {len(unlisted) - 5} more" if len(unlisted) > 5 else "")
            + "; remove files that are not part of the package"
        )
    if "conversion.json" not in sums:
        raise ValueError("Required file not listed in SHA256SUMS.json: conversion.json")
    kind = detect(root)
    adapter_kind = kind.endswith("adapter")
    heads = json.loads((root / "heads.json").read_text()) if adapter_kind and "heads.json" in sums else None
    missing = sorted(_required_files(root, kind, heads) - set(sums))
    if missing:
        raise ValueError(f"Required files not listed in SHA256SUMS.json for a {kind} package: {', '.join(missing)}")
    if heads is not None:
        # heads.json records the digests of the files it was exported with; the manifest must agree.
        adapter = "adapter/adapter_model.safetensors" if kind == "llama-adapter" else "adapter_model.safetensors"
        for name, field in [(heads["weights_file"], "weights_sha256"), (adapter, "adapter_sha256")]:
            if field in heads and sums.get(name) != heads[field]:
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
    p.add_argument("--revision", help="revision of a Hugging Face repo id (default: main)")
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
    if a.shell and a.component != "backend":
        p.error(
            "--shell prints the vLLM backend command; start the API with "
            "'bielik-one frontend <package> --backend-url http://<host>:<port>'"
        )
    if a.max_num_seqs is not None and not 1 <= a.max_num_seqs <= 256:
        p.error("--max-num-seqs must be between 1 and 256")
    if a.kv_cache_gb is not None and not a.kv_cache_gb > 0:
        p.error("--kv-cache-gb must be positive")
    if a.max_body_mb is not None and not (math.isfinite(a.max_body_mb) and a.max_body_mb * 2**20 >= 1):
        p.error("--max-body-mb must be a finite size of at least one byte")
    try:
        root = resolve_package(a.package, a.revision)
    except ValueError as exc:
        p.error(str(exc))
    try:
        verify(root)
        kind = detect(root)
    except (ValueError, FileNotFoundError) as exc:
        p.error(f"{root}: {exc}")
    back_port, front_port = PORTS[kind]
    if a.component == "backend":
        if a.host not in LOOPBACK:
            p.error(
                "--host sets the frontend address; use --backend-host to bind the raw vLLM backend"
            )
        argv, extra = backend(
            root, kind, a.backend_host, a.port or back_port, a.fp8, a.eager
        )
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
    launch_env = {**LAUNCH_ENV, **extra}
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
                },
                indent=2,
            )
        )
        return 0
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    sys.exit(main())
