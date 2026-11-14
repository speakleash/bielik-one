import shlex
import unittest.mock
import hashlib, json, tempfile, unittest
from pathlib import Path
import io, contextlib

from bielik_one import launch


def package(kind, root):
    files = {
        "conversion.json": {
            "base_revision": "r1",
            "source_repo": "x/y",
            "threshold": 0.5,
        }
    }
    if kind == "llama-adapter":
        files["serving.json"] = {
            "base_model": "org/base",
            "base_revision": "r1",
            "base_alias": "bielik-base",
            "rank": 64,
            "kv_cache_memory_bytes": 1,
        }
        files["heads.json"] = {"base_revision": "r1", "transformer_rank": 64, "weights_file": "heads.safetensors"}
        files["adapter/adapter_config.json"] = {"r": 64}
        weights = ["adapter/adapter_model.safetensors", "heads.safetensors"]
    elif kind == "llama-merged":
        files["serving.json"] = {
            "general_alias": "bielik-general",
            "kv_cache_memory_bytes": 1,
        }
        files["config.json"] = {"architectures": ["LlamaForCausalLM"]}
        # Sharded, as the published Llama packages are.
        files["model.safetensors.index.json"] = {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}
        weights = ["model-00001.safetensors", "model-00002.safetensors"]
    elif kind == "qwen-adapter":
        files["heads.json"] = {"base_revision": "r1", "weights_file": "heads.safetensors"}
        files["adapter_config.json"] = {"r": 128}
        weights = ["adapter_model.safetensors", "heads.safetensors"]
    else:
        files["conversion.json"]["merged_model"] = True
        files["config.json"] = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
        weights = ["model.safetensors"]  # a single file; the published package is sharded like Llama's
    if kind.endswith("merged"):
        files["tokenizer.json"] = {"model": {}}
        files["tokenizer_config.json"] = {}
    for name, value in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(value))
    for name in weights:
        if not (root / name).exists():
            (root / name).write_bytes(b"weights:" + name.encode())
    (root / "calibration.json").write_text("{}")
    sums = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))


def dry(*args):
    """Run the launcher; a refused package (argparse error, exit 2) surfaces as ValueError with its message."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            launch.main([*args, "--dry-run"])
    except SystemExit as exc:
        raise ValueError(err.getvalue()) from exc
    return json.loads(out.getvalue())


class LaunchTests(unittest.TestCase):
    def test_all_kinds(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package(kind, root)
                back = dry("backend", d)
                front = dry("frontend", d)
                self.assertEqual(back["kind"], kind)
                cmd = front["command"]
                self.assertEqual(
                    cmd[cmd.index("--calibration") + 1], str(root / "calibration.json")
                )
                self.assertIn("--backend-url", cmd)
                self.assertEqual(
                    cmd[cmd.index("--backend-url") + 1],
                    f"http://127.0.0.1:{launch.PORTS[kind][0]}",
                )
                raw = dry("frontend", d, "--no-calibration")["command"]
                self.assertIn("--no-calibration", raw)
                self.assertNotIn("--calibration", raw)
                self.assertNotIn("--quantization", back["command"])  # bfloat16 by default
                self.assertIn("--quantization", dry("backend", d, "--fp8")["command"])
                self.assertEqual(
                    back["env"].get("VLLM_PLUGINS"),
                    "systemone_heads" if "adapter" in kind else "",
                )

    def test_adapter_package_template_is_passed_to_frontend(self):
        for kind in ("llama-adapter", "qwen-adapter"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "chat_template.jinja").write_text("PINNED")
                package(kind, root)
                cmd = dry("frontend", d)["command"]
                meta = json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text())
                self.assertEqual(
                    meta["chat_template_path"], str(root / "chat_template.jinja")
                )
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                cmd = dry("frontend", d)["command"]
                self.assertNotIn(
                    "chat_template_path",
                    json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text()),
                )

    def test_unlisted_or_missing_template_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            (Path(d) / "chat_template.jinja").write_text("STRAY")
            with self.assertRaises(ValueError):
                dry("frontend", d)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "chat_template.jinja").write_text("PINNED")
            package("llama-adapter", Path(d))
            (Path(d) / "chat_template.jinja").unlink()
            with self.assertRaisesRegex(ValueError, "Missing file"):
                dry("frontend", d)

    def test_heads_manifest_cross_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "heads.safetensors").write_bytes(b"heads")
            package("llama-adapter", root)
            heads = json.loads((root / "heads.json").read_text())
            heads["weights_sha256"] = "0" * 64
            (root / "heads.json").write_text(json.dumps(heads))
            sums = json.loads((root / "SHA256SUMS.json").read_text())
            sums["heads.json"] = hashlib.sha256(
                (root / "heads.json").read_bytes()
            ).hexdigest()
            (root / "SHA256SUMS.json").write_text(json.dumps(sums))
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                dry("backend", d)

    def test_routed_adapter_package(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-adapter", root)
            serving = json.loads((root / "serving.json").read_text())
            serving["base_row_types"] = ["choice", "noul", "score"]
            (root / "serving.json").write_text(json.dumps(serving))
            heads = json.loads((root / "heads.json").read_text())

            def resum():
                sums = {
                    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*")
                    if p.is_file() and p.name != "SHA256SUMS.json"
                }
                (root / "SHA256SUMS.json").write_text(json.dumps(sums))

            resum()
            with self.assertRaisesRegex(ValueError, "base_row_heads"):
                dry("backend", d)
            heads["base_row_heads"] = True
            (root / "heads.json").write_text(json.dumps(heads))
            resum()
            back = dry("backend", d)["command"]
            self.assertEqual(
                back[back.index("--logits-processors") + 1],
                "systemone_vllm_heads.processor:SystemOneBaseHeadsProcessor",
            )
            cmd = dry("frontend", d)["command"]
            meta = json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text())
            self.assertEqual(meta["base_row_types"], ["choice", "noul", "score"])
            self.assertEqual(meta["base_alias"], "bielik-base")

    def test_checksum_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            (Path(d) / "calibration.json").write_text('{"changed": 1}')
            with self.assertRaises(ValueError):
                dry("frontend", d)

    def test_auxiliary_files_can_be_edited_or_omitted(self):
        names = ("README.md", "LICENSE", ".gitattributes", "requirements.txt")
        for kind in launch.PORTS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                for name in names:
                    (root / name).write_text("published content")
                package(kind, root)
                published = json.loads((root / "SHA256SUMS.json").read_text())
                for name in names:
                    (root / name).write_text("updated content")
                for component in ("backend", "frontend"):
                    self.assertEqual(dry(component, d)["kind"], kind)
                for name in names:
                    (root / name).unlink()
                for component in ("backend", "frontend"):
                    self.assertEqual(dry(component, d)["kind"], kind)
                self.assertEqual(launch.verify(root), published)

    def test_all_weight_files_remain_required_and_checksummed(self):
        names = (
            "model-00001.safetensors",
            "adapter/adapter_model.safetensors",
            "heads.safetensors",
        )
        for kind in launch.PORTS:
            for name in names:
                with self.subTest(kind=kind, file=name), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    package(kind, root)
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"published weights")
                    sums = json.loads((root / "SHA256SUMS.json").read_text())
                    sums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                    (root / "SHA256SUMS.json").write_text(json.dumps(sums))
                    for component in ("backend", "frontend"):
                        self.assertEqual(dry(component, d)["kind"], kind)
                    target.write_bytes(b"corrupted weights")
                    for component in ("backend", "frontend"):
                        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                            dry(component, d)
                    target.unlink()
                    for component in ("backend", "frontend"):
                        with self.assertRaisesRegex(ValueError, "Missing file"):
                            dry(component, d)

    def test_runtime_metadata_remains_checksummed(self):
        names = (
            "config.json", "conversion.json", "serving.json", "calibration.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        )
        for name in names:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package("llama-merged", root)
                target = root / name
                if not target.exists():
                    target.write_text("published content")
                sums = json.loads((root / "SHA256SUMS.json").read_text())
                sums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                (root / "SHA256SUMS.json").write_text(json.dumps(sums))
                target.write_text("corrupted content")
                for component in ("backend", "frontend"):
                    with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                        dry(component, d)


def edit_manifest(root, drop=(), **digests):
    """Rewrite SHA256SUMS.json without the `drop` entries and with any digests given by keyword (dots as `__`)."""
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    for name in drop:
        sums.pop(name)
    sums.update({name.replace("__", "."): value for name, value in digests.items()})
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))


def resign(root, name):
    """Re-record the digest of `name` after a test has changed it."""
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    sums[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))


class ManifestCompletenessTests(unittest.TestCase):
    """A package passes only when every file it ships is listed and its kind's runtime files are among them."""

    def refused(self, d, pattern):
        for component in ("backend", "frontend"):
            with self.assertRaisesRegex(ValueError, pattern):
                dry(component, d)

    def test_empty_or_malformed_manifest_is_refused(self):
        for kind in launch.PORTS:
            for manifest in ("{}", "[]", '"x"'):
                with self.subTest(kind=kind, manifest=manifest), tempfile.TemporaryDirectory() as d:
                    package(kind, Path(d))
                    (Path(d) / "SHA256SUMS.json").write_text(manifest)
                    self.refused(d, "must be a non-empty object")

    def test_malformed_digest_is_refused(self):
        for digest in ("0" * 63, "0" * 64 + "0", "G" * 64, "A" * 64, 5, None):
            with self.subTest(digest=digest), tempfile.TemporaryDirectory() as d:
                package("llama-merged", Path(d))
                edit_manifest(Path(d), config__json=digest)
                self.refused(d, "Invalid SHA-256 digest")

    def test_shipped_file_without_an_entry_is_refused(self):
        # The file is on disk but its entry is gone: today it would be loaded unverified.
        cases = {
            "llama-merged": ["config.json", "tokenizer.json", "model-00001.safetensors", "model.safetensors.index.json"],
            "qwen-merged": ["config.json", "tokenizer_config.json", "model.safetensors"],
            "llama-adapter": ["serving.json", "heads.json", "heads.safetensors", "adapter/adapter_model.safetensors",
                              "adapter/adapter_config.json", "calibration.json"],
            "qwen-adapter": ["heads.json", "heads.safetensors", "adapter_model.safetensors", "adapter_config.json"],
        }
        for kind, names in cases.items():
            for name in names:
                with self.subTest(kind=kind, file=name), tempfile.TemporaryDirectory() as d:
                    package(kind, Path(d))
                    edit_manifest(Path(d), drop=[name])
                    self.refused(d, f"Not listed in SHA256SUMS.json: {name}")

    def test_required_file_missing_from_package_and_manifest_is_refused(self):
        cases = {
            "llama-merged": ["config.json", "tokenizer.json", "model-00002.safetensors"],
            "qwen-merged": ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
            "llama-adapter": ["heads.safetensors", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"],
            "qwen-adapter": ["heads.safetensors", "adapter_config.json"],
        }
        for kind, names in cases.items():
            for name in names:
                with self.subTest(kind=kind, file=name), tempfile.TemporaryDirectory() as d:
                    package(kind, Path(d))
                    (Path(d) / name).unlink()
                    edit_manifest(Path(d), drop=[name])
                    self.refused(d, f"Required files not listed in SHA256SUMS.json for a {kind} package: .*{name}")

    def test_conversion_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            package("qwen-merged", Path(d))
            (Path(d) / "conversion.json").unlink()
            edit_manifest(Path(d), drop=["conversion.json"])
            self.refused(d, "Required file not listed in SHA256SUMS.json: conversion.json")

    def test_unsharded_merged_package_needs_its_single_weights_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-merged", root)
            for name in ("model.safetensors.index.json", "model-00001.safetensors", "model-00002.safetensors"):
                (root / name).unlink()
            edit_manifest(root, drop=["model.safetensors.index.json", "model-00001.safetensors", "model-00002.safetensors"])
            self.refused(d, "Required files .*model.safetensors")
            (root / "model.safetensors").write_bytes(b"weights")
            resign(root, "model.safetensors")
            self.assertEqual(dry("backend", d)["kind"], "llama-merged")

    def test_index_without_weight_map_or_with_an_escaping_shard_is_refused(self):
        for index, pattern in (({}, "no weight_map"), ({"weight_map": {"a": "../x.safetensors"}}, "Invalid checksum path")):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package("llama-merged", root)
                (root / "model.safetensors.index.json").write_text(json.dumps(index))
                resign(root, "model.safetensors.index.json")
                self.refused(d, pattern)

    def test_heads_without_weights_file_is_refused(self):
        for kind in ("llama-adapter", "qwen-adapter"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package(kind, root)
                heads = json.loads((root / "heads.json").read_text())
                del heads["weights_file"]
                (root / "heads.json").write_text(json.dumps(heads))
                resign(root, "heads.json")
                self.refused(d, "heads.json must name its weights_file")

    def test_heads_digests_must_match_the_manifest_for_both_adapter_layouts(self):
        adapter = {"llama-adapter": "adapter/adapter_model.safetensors", "qwen-adapter": "adapter_model.safetensors"}
        for kind, adapter_file in adapter.items():
            for field, name in (("weights_sha256", "heads.safetensors"), ("adapter_sha256", adapter_file)):
                with self.subTest(kind=kind, field=field), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    package(kind, root)
                    heads = json.loads((root / "heads.json").read_text())
                    sums = json.loads((root / "SHA256SUMS.json").read_text())
                    heads.update(weights_sha256=sums["heads.safetensors"], adapter_sha256=sums[adapter_file])
                    (root / "heads.json").write_text(json.dumps(heads))
                    resign(root, "heads.json")
                    self.assertEqual(dry("backend", d)["kind"], kind)  # consistent digests pass
                    heads[field] = "0" * 64
                    (root / "heads.json").write_text(json.dumps(heads))
                    resign(root, "heads.json")
                    self.refused(d, f"Manifest mismatch: {name}")

    def test_hidden_hugging_face_metadata_is_ignored(self):
        for kind in launch.PORTS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package(kind, root)
                for name in (".cache/huggingface/download/config.json.metadata", ".eval_results/typed-decisions.yaml",
                             ".DS_Store"):
                    (root / name).parent.mkdir(parents=True, exist_ok=True)
                    (root / name).write_text("not part of the package")
                for component in ("backend", "frontend"):
                    self.assertEqual(dry(component, d)["kind"], kind)

    def test_unlisted_extra_runtime_file_is_refused(self):
        for name in ("generation_config.json", "special_tokens_map.json", "model-00003.safetensors", "adapter/extra.bin"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as d:
                package("llama-adapter", Path(d))
                (Path(d) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / name).write_text("{}")
                self.refused(d, f"Not listed in SHA256SUMS.json: {name}")

    def test_middleware_package_check_uses_the_same_rules(self):
        with tempfile.TemporaryDirectory() as d:
            package("qwen-adapter", Path(d))
            launch.verify(Path(d))
            (Path(d) / "SHA256SUMS.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "must be a non-empty object"):
                launch.verify(Path(d))


if __name__ == "__main__":
    unittest.main()


class PackageSpecTests(unittest.TestCase):
    def test_local_dir_repo_id_and_errors(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(launch.resolve_package(d), Path(d).resolve())
            with self.assertRaises(ValueError):
                launch.resolve_package(d, "abc")
            with mock.patch("huggingface_hub.snapshot_download", return_value=d) as dl:
                self.assertEqual(launch.resolve_package("org/Model", "rev1"), Path(d))
                dl.assert_called_once_with("org/Model", revision="rev1")
        with self.assertRaises(ValueError):
            launch.resolve_package("./missing-dir")

    def test_hf_cache_layout_with_symlinks_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            package("llama-merged", src)
            blobs = Path(d) / "blobs"
            blobs.mkdir()
            snap = Path(d) / "snapshots" / "rev"
            snap.mkdir(parents=True)
            for f in src.iterdir():
                (blobs / f.name).write_bytes(f.read_bytes())
                (snap / f.name).symlink_to(Path("../..") / "blobs" / f.name)
            launch.verify(snap)
            self.assertEqual(dry("backend", str(snap))["kind"], "llama-merged")

    def test_checksum_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-merged", root)
            (root / "SHA256SUMS.json").write_text(json.dumps({"../x": "0"}))
            with self.assertRaises(ValueError):
                launch.verify(root)

    def test_auxiliary_names_do_not_bypass_path_validation(self):
        for name in ("../README.md", "/LICENSE", "..\\requirements.txt"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "SHA256SUMS.json").write_text(json.dumps({name: "0" * 64}))
                with self.assertRaisesRegex(ValueError, "Invalid checksum path"):
                    launch.verify(root)

    def test_auxiliary_exclusions_apply_only_at_package_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "adapter" / "README.md"
            target.parent.mkdir()
            target.write_text("changed")
            (root / "SHA256SUMS.json").write_text(
                json.dumps({"adapter/README.md": "0" * 64})
            )
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                launch.verify(root)


class EagerFlagTests(unittest.TestCase):
    def test_graphs_by_default_eager_on_request(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                self.assertNotIn("--enforce-eager", dry("backend", d)["command"], kind)
                self.assertIn("--enforce-eager", dry("backend", d, "--eager")["command"], kind)


class ProfileOverrideTests(unittest.TestCase):
    def test_max_num_seqs_and_kv_cache_override(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                base = dry("backend", d)["command"]
                cmd = dry("backend", d, "--max-num-seqs", "32", "--kv-cache-gb", "4")["command"]
                self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "32", kind)
                self.assertEqual(cmd[cmd.index("--kv-cache-memory-bytes") + 1], str(4 * 2**30), kind)
                # nothing else changes
                strip = lambda c: [x for i, x in enumerate(c) if i == 0 or c[i - 1] not in ("--max-num-seqs", "--kv-cache-memory-bytes")]
                self.assertEqual(len(strip(cmd)), len(strip(base)))
                front = dry("frontend", d, "--max-num-seqs", "64")["command"]
                self.assertEqual(front[front.index("--workers") + 1], "64", kind)

    def test_max_body_mb_goes_to_the_frontend(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                self.assertNotIn("--max-body-mb", dry("frontend", d)["command"])
                front = dry("frontend", d, "--max-body-mb", "4")["command"]
                self.assertEqual(front[front.index("--max-body-mb") + 1], "4.0", kind)

    def test_invalid_values_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            for args in (["--max-num-seqs", "0"], ["--kv-cache-gb", "-1"], ["--max-body-mb", "0"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launch.main(["backend", d, *args, "--dry-run"])


class PrintedCommandTests(unittest.TestCase):
    """What --dry-run and --shell print is exactly what the launcher would execute, environment included."""

    def shell(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(launch.main([*args, "--shell"]), 0)
        return shlex.split(out.getvalue())

    def test_dry_run_env_includes_the_launcher_environment(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                env = dry("backend", d)["env"]
                for key, value in launch.LAUNCH_ENV.items():
                    self.assertEqual(env[key], value, kind)
                if kind.endswith("adapter"):
                    self.assertEqual(env["VLLM_PLUGINS"], "systemone_heads")

    def test_shell_line_matches_the_dry_run(self):
        for kind in launch.PORTS:
            for component in ("backend",):
                with tempfile.TemporaryDirectory() as d:
                    package(kind, Path(d))
                    args = (component, d, "--max-num-seqs", "16", "--port", "9100")
                    expected = dry(*args)
                    words = self.shell(*args)
                    self.assertEqual(words[0], "env")
                    n = len(expected["env"])
                    self.assertEqual(dict(w.split("=", 1) for w in words[1 : 1 + n]), expected["env"])
                    self.assertEqual(words[1 + n :], expected["command"], (kind, component))

    def test_shell_starts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            with unittest.mock.patch.object(launch.os, "execvpe") as run:
                self.shell("backend", d)
            run.assert_not_called()

    def test_shell_is_backend_only(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                launch.main(["frontend", d, "--shell"])
            self.assertIn("--backend-url", err.getvalue())
