import math
from types import SimpleNamespace
import unittest

import httpx

from bielik_one.api.remote_vllm import BackendReadError, RemoteVllmEngine
from bielik_one.api.server import RequestError, SystemOneIn


class RemoteVllmTests(unittest.TestCase):
    def make_engine(self, broken=None):
        self.requests = []
        anchor = 1500
        metadata = dict(
            choice_capacity=255,
            labels_capacity=248,
            label_ids=list(range(1000, 1248)),
            anchor_id=anchor,
            bias=[0.7] + [-0.2] * 247,
            threshold=0.5,
            source_repo="test/decision",
        )

        def handler(request):
            import json

            body = json.loads(request.content)
            self.requests.append(body)
            ids = body["logprob_token_ids"]
            self.assertLessEqual(len(ids), 128)
            self.assertNotIn("allowed_token_ids", body)
            # A different whole-vocabulary normalizer on every read cancels
            # only when each chunk is compared with its own common anchor.
            normalizer = 100 + len(self.requests) * 30
            values = {
                f"token_id:{i}": (0 if i == anchor else (i % 11 - 5) * 0.25)
                - normalizer
                for i in ids
            }
            if broken == "missing":
                values.pop(f"token_id:{anchor}")
            if broken == "sentinel":
                values[f"token_id:{ids[0]}"] = -9999
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": len(body["prompt"])},
                    "choices": [{"logprobs": {"top_logprobs": [values]}}],
                },
            )

        engine = RemoteVllmEngine(
            metadata,
            "http://backend",
            tokenizer=object(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(engine.close)
        return engine

    def test_labels_248_keep_bias_and_chunk_anchor(self):
        engine = self.make_engine()
        ids = engine.parking["option_ids"]
        table, tokens = engine._read([10, 20, 30], ids, "labels")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(tokens, 3)  # prompt tokens once, not once per chunk
        for i, token in enumerate(ids):
            self.assertAlmostEqual(
                table[token] - table[1500], (token % 11 - 5) * 0.25 + engine.bias[i]
            )

    def test_choice_255_normalizes_all_chunks_together(self):
        engine = self.make_engine()
        ids = list(range(2000, 2255))
        table, tokens = engine._read([10, 20], ids, "choice")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(tokens, 2)  # prompt tokens once, not once per chunk
        denominator = sum(math.exp((i % 11 - 5) * 0.25) for i in ids)
        for i in ids:
            self.assertAlmostEqual(
                math.exp(table[i]), math.exp((i % 11 - 5) * 0.25) / denominator
            )
        self.assertAlmostEqual(sum(math.exp(x) for x in table.values()), 1)

    def test_missing_anchor_fails_instead_of_fabricating_probability(self):
        with self.assertRaises(BackendReadError):
            self.make_engine("missing")._read([10], [1000], "labels")

    def test_finite_negative_infinity_sentinel_is_rejected(self):
        with self.assertRaises(BackendReadError):
            self.make_engine("sentinel")._read([10], [1000], "labels")

    def test_labels_249_rejected_before_backend_call(self):
        engine = self.make_engine()
        row = SimpleNamespace(
            id="tags", question=SimpleNamespace(type="labels"), options=list(range(249))
        )
        with self.assertRaises(RequestError):
            engine.distributions([row])
        self.assertEqual(self.requests, [])

    def test_unknown_model_rejected_before_backend_call(self):
        engine = self.make_engine()
        body = SystemOneIn(
            model="qwen-base",
            state="x",
            questions={"x": {"type": "choice", "criteria": {"a": "a", "b": "b"}}},
        )
        with self.assertRaises(RequestError):
            engine.evaluate(body)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()


class PinnedTemplateTests(unittest.TestCase):
    def test_metadata_template_overrides_base_tokenizer(self):
        import tempfile
        from pathlib import Path

        tok = SimpleNamespace(chat_template="BASE")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text("PINNED {{ bos_token }}")
            metadata = dict(
                choice_capacity=255,
                labels_capacity=1,
                label_ids=[1000],
                anchor_id=1500,
                bias=[0.0],
                threshold=0.5,
                chat_template_path=str(path),
            )
            engine = RemoteVllmEngine(
                metadata, "http://backend", tokenizer=tok, client=httpx.Client()
            )
            engine.close()
        self.assertEqual(tok.chat_template, "PINNED {{ bos_token }}")
        tok2 = SimpleNamespace(chat_template="BASE")
        metadata.pop("chat_template_path")
        RemoteVllmEngine(
            metadata, "http://backend", tokenizer=tok2, client=httpx.Client()
        ).close()
        self.assertEqual(tok2.chat_template, "BASE")


class CalibrationTests(unittest.TestCase):
    TEMPS = {"choice": 2.0, "binary": 1.5, "score": 3.0, "labels": 0.8}

    def write(self, directory, temperatures):
        import json

        path = directory / "calibration.json"
        path.write_text(json.dumps({"temperatures": temperatures}))
        return path

    def test_load_maps_binary_to_noul_for_one_order_only(self):
        import tempfile
        from pathlib import Path
        from bielik_one.api.remote_vllm import load_calibration

        with tempfile.TemporaryDirectory() as d:
            table = load_calibration(self.write(Path(d), self.TEMPS))
        self.assertEqual(
            table, {"1": {"choice": 2.0, "noul": 1.5, "score": 3.0, "labels": 0.8}}
        )

    def test_invalid_files_are_rejected(self):
        import tempfile
        from pathlib import Path
        from bielik_one.api.remote_vllm import load_calibration

        bad = [
            {"choice": 2.0},
            {**self.TEMPS, "score": 0},
            {**self.TEMPS, "labels": True},
            {**self.TEMPS, "binary": float("nan")},
            {**self.TEMPS, "noul": 1.0},
        ]
        with tempfile.TemporaryDirectory() as d:
            for temperatures in bad:
                with self.assertRaises(ValueError):
                    load_calibration(self.write(Path(d), temperatures))

    def test_default_explicit_and_disabled_resolution(self):
        import tempfile
        from pathlib import Path
        from bielik_one.api.remote_vllm import resolve_calibration

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            metadata = d / "conversion.local.json"
            self.assertEqual(resolve_calibration(metadata, None, False), ({}, None))
            path = self.write(d, self.TEMPS)
            table, source = resolve_calibration(metadata, None, False)
            self.assertEqual(table["1"]["noul"], 1.5)
            self.assertEqual(source, str(path))
            self.assertEqual(resolve_calibration(metadata, None, True), ({}, None))
            other = d / "x"
            other.mkdir()
            explicit = self.write(other, {**self.TEMPS, "choice": 4.0})
            self.assertEqual(
                resolve_calibration(metadata, explicit, False)[0]["1"]["choice"], 4.0
            )

    def evaluate(self, calibration, **request):
        """Full _evaluate_once path with a synthetic backend distribution for one labels question."""
        from bielik_one.api.remote_vllm import load_calibration
        import tempfile
        from pathlib import Path

        engine = RemoteVllmTests.make_engine(
            SimpleNamespace(
                requests=[],
                assertLessEqual=lambda *a: None,
                assertNotIn=lambda *a: None,
                addCleanup=lambda f: None,
            )
        )
        with tempfile.TemporaryDirectory() as d:
            engine.temperatures = (
                load_calibration(self.write(Path(d), self.TEMPS)) if calibration else {}
            )
        anchor = engine.parking["anchor_id"]
        ids = engine.parking["option_ids"]
        engine.distributions = lambda prompts, **kw: (
            [{anchor: 0.0, ids[0]: 2.0 - 0.7, ids[1]: -1.0 + 0.2} for _ in prompts],
            7,
        )
        body = SystemOneIn(
            state="s",
            model="decision",
            questions={
                "q": {
                    "type": "labels",
                    "instructions": "Pick tags.",
                    "criteria": {"a": "first", "b": "second"},
                }
            },
            **request,
        )
        try:
            return engine._evaluate_once(body)["answers"]["q"]["probabilities"]
        finally:
            engine.close()

    def test_labels_probabilities_use_calibrated_default_and_request_override(self):
        raw = self.evaluate(False)
        cal = self.evaluate(True)
        logit = lambda p: math.log(p / (1 - p))
        for key in raw:
            self.assertAlmostEqual(logit(cal[key]), logit(raw[key]) / 0.8, places=5)
            self.assertEqual(cal[key] >= 0.5, raw[key] >= 0.5)
        self.assertEqual(self.evaluate(True, temperature=1.0), raw)
        # The one-order temperatures apply to every request mode, including two option orders.
        raw2, cal2 = self.evaluate(False, permutations=2), self.evaluate(True, permutations=2)
        for key in raw2:
            self.assertAlmostEqual(logit(cal2[key]), logit(raw2[key]) / 0.8, places=5)


class CalibrationReviewTests(unittest.TestCase):
    TEMPS = {"choice": 6.0, "binary": 6.0, "score": 6.0, "labels": 6.0}

    def engine(self, calibrated, threshold=0.5):
        from bielik_one.api.remote_vllm import load_calibration
        import json, tempfile
        from pathlib import Path

        table = {}
        if calibrated:
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "calibration.json"
                path.write_text(
                    json.dumps({"model": "test/decision", "temperatures": self.TEMPS})
                )
                table = load_calibration(path, "test/decision")
        metadata = dict(
            choice_capacity=255,
            labels_capacity=2,
            label_ids=[1000, 1001],
            anchor_id=1500,
            bias=[0.0, 0.0],
            threshold=threshold,
            source_repo="test/decision",
        )
        engine = RemoteVllmEngine(
            metadata,
            "http://backend",
            tokenizer=object(),
            calibration=table,
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(500))
            ),
        )
        self.addCleanup(engine.close)
        # Two-option choice: 97% on the first option, read from a synthetic backend table.
        engine.distributions = lambda prompts, **kw: (
            [{11: math.log(0.97), 12: math.log(0.03)} for _ in prompts],
            5,
        )
        return engine

    def body(self, **extra):
        return SystemOneIn(
            state="s",
            model="decision",
            questions={
                "q": {
                    "type": "choice",
                    "instructions": "Pick.",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
            **extra,
        )

    def run_choice(self, engine, **extra):
        from unittest.mock import patch

        with patch(
            "bielik_one.api.server.answer_token_ids",
            lambda tok, row, semantic: [11, 12],
        ):
            return engine.evaluate(self.body(**extra))

    def test_choice_is_calibrated_and_decision_kept(self):
        raw = self.run_choice(self.engine(False))["answers"]["q"]
        cal = self.run_choice(self.engine(True))["answers"]["q"]
        self.assertEqual(raw["choice"], cal["choice"])
        self.assertAlmostEqual(raw["probabilities"]["a"], 0.97, places=5)
        self.assertAlmostEqual(raw["confidence"], 2 * 0.97 - 1, places=5)  # two options: (p - 1/2) / (1/2)
        expected = 0.97 ** (1 / 6) / (0.97 ** (1 / 6) + 0.03 ** (1 / 6))
        self.assertAlmostEqual(cal["probabilities"]["a"], expected, places=5)
        self.assertAlmostEqual(cal["confidence"], 2 * expected - 1, places=5)

    def test_auto_reasoning_gate_uses_raw_confidence(self):
        for calibrated in (False, True):
            engine = self.engine(calibrated)
            engine.reasoning_enabled = True  # disabled in 1.0; the gate stays tested for the rebuild
            calls = []
            engine.reasoners["original"] = lambda row, n: (
                calls.append(row.id)
                or {"text": "x", "input_tokens": 1, "output_tokens": 1}
            )
            # Raw confidence 0.97 >= 0.9, calibrated would be ~0.64 < 0.9: must not reason in either case.
            out = self.run_choice(engine, reasoning="auto", reasoning_threshold=0.9)
            self.assertEqual(calls, [], calibrated)
            self.assertIn("q", out["answers"])

    def test_reasoning_disabled_in_this_release(self):
        with self.assertRaises(RequestError) as ctx:
            self.run_choice(self.engine(False), reasoning="on")
        self.assertEqual(ctx.exception.loc, ["body", "reasoning"])
        for value in ("auto", "always"):
            with self.assertRaises(RequestError):
                self.run_choice(self.engine(False), reasoning=value)
        with self.assertRaises(Exception):
            self.run_choice(self.engine(False), reasoning="on", reasoning_model="lora")

    def test_wrong_model_and_threshold_rejected(self):
        from bielik_one.api.remote_vllm import load_calibration
        import json, tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.json"
            path.write_text(
                json.dumps({"model": "other/model", "temperatures": self.TEMPS})
            )
            with self.assertRaises(ValueError):
                load_calibration(path, "test/decision")
        with self.assertRaises(ValueError):
            self.engine(True, threshold=0.4)
        self.engine(False, threshold=0.4)

    def test_main_entrypoints_pass_calibration(self):
        import importlib.util

        if importlib.util.find_spec("bielik_one.api.merged_vllm") is None:
            self.skipTest("merged frontend not shipped in this package")
        import json, sys, tempfile
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import patch
        import bielik_one.api.remote_vllm as remote
        import bielik_one.api.merged_vllm as merged

        seen, metas, apps = [], [], []

        class FakeEngine:
            def __init__(self, metadata, url, **kwargs):
                seen.append(kwargs)
                metas.append(metadata)
                self.reasoners = {}

        uvicorn = ModuleType("uvicorn")
        uvicorn.run = lambda *a, **k: None
        with tempfile.TemporaryDirectory() as d:
            # HF-cache layout: snapshot entries are symlinks into blobs/ (files must be found next to the link)
            blobs, d = Path(d) / "blobs", Path(d) / "snapshot"
            blobs.mkdir()
            d.mkdir()
            meta = d / "conversion.local.json"
            (blobs / "b1").write_text(
                json.dumps(
                    {
                        "source_repo": "test/decision",
                        "output_id_map": {},
                        "chat_blocked_tokens": [],
                    }
                )
            )
            meta.symlink_to(blobs / "b1")
            (blobs / "b2").write_text(
                json.dumps({"model": "test/decision", "temperatures": self.TEMPS})
            )
            (d / "calibration.json").symlink_to(blobs / "b2")
            with (
                patch.dict(sys.modules, {"uvicorn": uvicorn}),
                patch.object(remote, "RemoteVllmEngine", FakeEngine),
                patch.object(remote, "build_remote_app", lambda e, **k: apps.append(k)),
                patch.object(sys, "argv", ["x", "--metadata", str(meta)]),
            ):
                remote.main()
            with (
                patch.object(merged, "uvicorn", uvicorn),
                patch.object(merged, "RemoteVllmEngine", FakeEngine),
                patch.object(merged, "build_remote_app", lambda e, **k: apps.append(k)),
                patch.object(merged, "MaskChatTokens", lambda app, blocked: None),
                patch.object(sys, "argv", ["x", "--metadata", str(meta)]),
            ):
                merged.main()
            with (
                patch.dict(sys.modules, {"uvicorn": uvicorn}),
                patch.object(remote, "RemoteVllmEngine", FakeEngine),
                patch.object(remote, "build_remote_app", lambda e, **k: apps.append(k)),
                patch.object(
                    sys, "argv", ["x", "--metadata", str(meta), "--no-calibration"]
                ),
            ):
                remote.main()
        self.assertEqual(
            [k["calibration"]["1"]["choice"] for k in seen[:2]], [6.0, 6.0]
        )
        self.assertEqual(seen[2]["calibration"], {})
        self.assertEqual(metas[1]["base_snapshot"], str(d))
        self.assertEqual(
            [k["chat_model"] for k in apps], ["qwen-base", "qwen-general", "qwen-base"]
        )


class RoutedBaseRowTests(unittest.TestCase):
    def test_categorical_types_go_to_base_alias_with_flag_labels_to_adapter(self):
        seen = []

        def handler(request):
            import json

            body = json.loads(request.content)
            seen.append(body)
            ids = body["logprob_token_ids"]
            values = {f"token_id:{i}": (-9999.0 if i == 0 else -1.0) for i in ids}
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": len(body["prompt"])},
                    "choices": [{"logprobs": {"top_logprobs": [values]}}],
                },
            )

        metadata = dict(
            choice_capacity=255,
            labels_capacity=2,
            label_ids=[1000, 1001],
            anchor_id=1500,
            bias=[0.0, 0.0],
            threshold=0.5,
            base_row_types=["choice", "noul", "score"],
            base_alias="bielik-base",
        )
        engine = RemoteVllmEngine(
            metadata,
            "http://backend",
            tokenizer=object(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(engine.close)
        engine._read([1, 2], [11, 12], "choice")
        engine._read([1, 2], [1000], "labels")
        self.assertIn(engine.canary_id, seen[0]["logprob_token_ids"])
        self.assertNotIn(engine.canary_id, seen[1]["logprob_token_ids"])
        self.assertEqual(seen[0]["model"], "bielik-base")
        self.assertEqual(seen[0]["vllm_xargs"], {"systemone_head": "base"})
        self.assertEqual(seen[1]["model"], "decision")
        self.assertNotIn("vllm_xargs", seen[1])

    def test_routing_requires_base_alias(self):
        metadata = dict(
            choice_capacity=255,
            labels_capacity=1,
            label_ids=[1000],
            anchor_id=1500,
            bias=[0.0],
            threshold=0.5,
            base_row_types=["choice"],
        )
        with self.assertRaises(ValueError):
            RemoteVllmEngine(
                metadata, "http://backend", tokenizer=object(), client=httpx.Client()
            )

    def test_missing_processor_detected_by_canary(self):
        def handler(request):
            import json

            body = json.loads(request.content)
            values = {
                f"token_id:{i}": -1.0 for i in body["logprob_token_ids"]
            }  # canary finite: no processor
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": len(body["prompt"])},
                    "choices": [{"logprobs": {"top_logprobs": [values]}}],
                },
            )

        metadata = dict(
            choice_capacity=255,
            labels_capacity=2,
            label_ids=[1000, 1001],
            anchor_id=1500,
            bias=[0.0, 0.0],
            threshold=0.5,
            base_row_types=["choice"],
            base_alias="bielik-base",
        )
        engine = RemoteVllmEngine(
            metadata,
            "http://backend",
            tokenizer=object(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(engine.close)
        with self.assertRaises(Exception):
            engine._read([1, 2], [11, 12], "choice")


class PublicProxyTests(unittest.TestCase):
    def test_strips_internal_flags_and_rejects_prompt_logprobs(self):
        import json
        from bielik_one.api.remote_vllm import public_request_body

        out = json.loads(
            public_request_body(
                json.dumps(
                    {
                        "model": "decision",
                        "vllm_xargs": {"systemone_head": "base"},
                        "prompt": "x",
                    }
                ).encode()
            )
        )
        self.assertNotIn("vllm_xargs", out)
        raw = json.dumps({"model": "bielik-base", "prompt": "x"}).encode()
        self.assertIs(public_request_body(raw), raw)
        for bad in (
            {"prompt": "x", "prompt_logprobs": 1},
            {"prompt": "x", "echo": True, "logprobs": 1},
        ):
            with self.assertRaises(RequestError):
                public_request_body(json.dumps(bad).encode())
        self.assertEqual(public_request_body(b"not json"), b"not json")

    def test_rejects_decision_models_for_text_generation(self):
        import json
        from bielik_one.api.remote_vllm import public_request_body

        names = {"decision", "test/decision"}
        for model in names:
            with self.assertRaises(RequestError) as ctx:
                public_request_body(
                    json.dumps({"model": model, "prompt": "x"}).encode(),
                    names,
                    "bielik-base",
                )
            self.assertIn("bielik-base", ctx.exception.msg)
        for body in ({"model": "bielik-base", "prompt": "x"}, {"model": ["decision"]}):
            raw = json.dumps(body).encode()
            self.assertIs(public_request_body(raw, names, "bielik-base"), raw)
        filled = json.loads(
            public_request_body(json.dumps({"prompt": "x", "vllm_xargs": {}}).encode(), names, "bielik-base")
        )
        self.assertEqual(filled, {"prompt": "x", "model": "bielik-base"})
        for body in ({"prompt": "x", "prompt_logprobs": 1}, {"model": "", "prompt": "x", "echo": True, "logprobs": 1}):
            with self.assertRaises(RequestError):
                public_request_body(json.dumps(body).encode(), names, "bielik-base")
        for body in (
            {"model": "bielik-base", "prompt": "x", "echo": True, "logprobs": 0},
            {"model": "bielik-base", "messages": [], "echo": True, "top_logprobs": 0},
        ):
            with self.assertRaises(RequestError):
                public_request_body(json.dumps(body).encode(), names, "bielik-base")
        empty = json.loads(public_request_body(b'{"model": "", "prompt": "x"}', names, "bielik-base"))
        self.assertEqual(empty["model"], "bielik-base")

    def test_models_listing_hides_decision_names(self):
        import json
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from bielik_one.api import remote_vllm

        listing = {"object": "list", "data": [{"id": "decision"}, {"id": "qwen-general"}]}

        class Engine:
            backend_url = "http://backend"
            aliases = {"decision", "test/decision"}
            temperatures = {}
            calibration_source = None

            def close(self):
                pass

        async def get(url, **kw):
            return httpx.Response(200, json=listing, request=httpx.Request("GET", url))

        app = remote_vllm.build_remote_app(Engine(), chat_model="qwen-general")
        with patch.object(httpx.AsyncClient, "get", lambda self, url, **kw: get(url)):
            with TestClient(app) as client:
                data = client.get("/v1/models").json()["data"]
        self.assertEqual([m["id"] for m in data], ["qwen-general"])


class ChunkWidthTests(unittest.TestCase):
    def test_wide_base_row_reads_stay_within_vllm_cap(self):
        import json

        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            ids = body["logprob_token_ids"]
            values = {f"token_id:{i}": (-9999.0 if i == 0 else -1.0) for i in ids}
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": len(body["prompt"])},
                    "choices": [{"logprobs": {"top_logprobs": [values]}}],
                },
            )

        engine = RemoteVllmEngine(
            dict(choice_capacity=255, labels_capacity=2, label_ids=[1000, 1001], anchor_id=1500,
                 bias=[0.0, 0.0], threshold=0.5, base_row_types=["choice", "noul", "score"],
                 base_alias="bielik-base"),
            "http://backend",
            tokenizer=object(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(engine.close)
        ids = list(range(2000, 2255))
        table, _ = engine._read([1, 2], ids, "choice")
        self.assertEqual(len(seen), 3)  # 126 + 126 + 3 options
        for body in seen:
            self.assertLessEqual(len(body["logprob_token_ids"]), 128)
            self.assertIn(engine.canary_id, body["logprob_token_ids"])
        self.assertAlmostEqual(sum(math.exp(table[i]) for i in ids), 1.0, places=6)


class PackageDirTests(unittest.TestCase):
    def test_link_directory_first_then_target_directory(self):
        import tempfile
        from pathlib import Path
        from bielik_one.api.remote_vllm import package_dir

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # HF cache: files are links next to the metadata link, blobs have hash names
            (d / "blobs").mkdir()
            (d / "snap").mkdir()
            for name, blob in (("conversion.json", "b1"), ("tokenizer_config.json", "b2")):
                (d / "blobs" / blob).write_text("{}")
                (d / "snap" / name).symlink_to(d / "blobs" / blob)
            self.assertEqual(package_dir(d / "snap" / "conversion.json", "tokenizer_config.json"), d / "snap")
            # metadata linked from elsewhere into a real package directory
            (d / "run").mkdir()
            (d / "run" / "conversion.json").symlink_to(d / "snap" / "conversion.json")
            self.assertEqual(package_dir(d / "run" / "conversion.json", "tokenizer_config.json"), d / "snap")


class ReasonerErrorTests(unittest.TestCase):
    def test_endpoint_errors_map_to_reasoning_error(self):
        import io
        from unittest.mock import patch
        from urllib.error import HTTPError, URLError
        from bielik_one.api import reasoning

        r = reasoning.EndpointReasoner("http://127.0.0.1:1/v1", "bielik-base")
        row = SimpleNamespace()
        with patch.object(reasoning, "reasoning_messages", lambda row, tok: []):
            for exc, status in (
                (HTTPError("u", 400, "bad", {}, io.BytesIO(b"too long")), 422),
                (HTTPError("u", 500, "bad", {}, io.BytesIO(b"")), 502),
                (URLError("refused"), 502),
            ):
                def fail(*a, exc=exc, **k):
                    raise exc

                with patch.object(reasoning._OPENER, "open", fail):
                    with self.assertRaises(reasoning.ReasoningError) as ctx:
                        r(row, 100)
                self.assertEqual(ctx.exception.status, status)

    def test_reasoner_ignores_proxy_env(self):
        from bielik_one.api import reasoning

        # ProxyHandler({}) replaces the env-reading default handler and registers no proxy routes
        self.assertFalse(any(getattr(h, "proxies", None) for h in reasoning._OPENER.handlers))


class ParallelReasoningTests(unittest.TestCase):
    def test_analyses_run_concurrently_in_question_order(self):
        import threading
        from bielik_one.api.server import LetterEngine

        rows = [SimpleNamespace(id=f"q{i}") for i in range(5)]
        barrier = threading.Barrier(5, timeout=5)  # deadlocks (BrokenBarrierError) unless all five run at once

        def reasoner(row, n):
            barrier.wait()
            return {"text": row.id, "input_tokens": 1, "output_tokens": n}

        out = LetterEngine._reason(SimpleNamespace(reasoning_parallel=8), reasoner, rows, 7)
        self.assertEqual(list(out), [r.id for r in rows])
        self.assertEqual(out["q3"], {"text": "q3", "input_tokens": 1, "output_tokens": 7})

    def test_first_failure_is_raised(self):
        from bielik_one.api.reasoning import ReasoningError
        from bielik_one.api.server import LetterEngine

        def reasoner(row, n):
            if row.id == "q1":
                raise ReasoningError("boom", 422)
            return {"text": "", "input_tokens": 0, "output_tokens": 0}

        rows = [SimpleNamespace(id=f"q{i}") for i in range(4)]
        with self.assertRaises(ReasoningError):
            LetterEngine._reason(SimpleNamespace(reasoning_parallel=2), reasoner, rows, 1)
