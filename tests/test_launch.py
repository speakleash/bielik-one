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
        files["heads.json"] = {"base_revision": "r1", "transformer_rank": 64}
    elif kind == "llama-merged":
        files["serving.json"] = {
            "general_alias": "bielik-general",
            "kv_cache_memory_bytes": 1,
        }
        files["config.json"] = {"architectures": ["LlamaForCausalLM"]}
    elif kind == "qwen-adapter":
        files["heads.json"] = {"base_revision": "r1"}
        (root / "adapter_model.safetensors").write_bytes(b"x")
    else:
        files["conversion.json"]["merged_model"] = True
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "calibration.json").write_text("{}")
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
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
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.iterdir()
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
