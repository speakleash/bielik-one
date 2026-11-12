"""CPU tests of the heads plugin core (no vLLM needed)."""

import sys, unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm-heads-plugin/src"))
from systemone_vllm_heads import core  # noqa: E402


class Dir(Enum):
    UNIDIRECTIONAL = 1
    SWAP = 2


def params(flag=None):
    return SimpleNamespace(extra_args={"systemone_head": flag} if flag else None)


def update(removed=(), added=(), moved=()):
    return SimpleNamespace(
        removed=list(removed),
        added=[(i, p, None, []) for i, p in added],
        moved=list(moved),
    )


def readout(base_row_heads=True):
    torch.manual_seed(0)
    w = torch.randn(4, 6)
    w[3] = 0  # last row = anchor
    return core.DecisionReadout(
        w,
        torch.tensor([2, 5, 7, 9]),
        vocab_size=12,
        anchor_id=9,
        base_row_heads=base_row_heads,
    )


class FlagTests(unittest.TestCase):
    def test_add_remove_move_swap(self):
        f = {}
        core.update_flags(
            f, update(added=[(0, params("base")), (1, params()), (2, params("base"))])
        )
        self.assertEqual(sorted(f), [0, 2])
        core.update_flags(f, update(removed=[0]))
        self.assertEqual(sorted(f), [2])
        core.update_flags(f, update(moved=[(2, 0, Dir.UNIDIRECTIONAL)]))
        self.assertEqual(sorted(f), [0])
        core.update_flags(
            f, update(added=[(1, params("base"))], moved=[(0, 3, Dir.SWAP)])
        )
        self.assertEqual(sorted(f), [1, 3])
        core.update_flags(
            f, update(added=[(3, params())])
        )  # slot reused by an unflagged request
        self.assertEqual(sorted(f), [1])
        self.assertFalse(core.update_flags(f, None))


class BaseHeadsTests(unittest.TestCase):
    def test_only_flagged_base_rows_change_and_decision_rows_unchanged(self):
        r = readout()
        hidden = torch.randn(3, 6)
        sampler = torch.tensor(
            [-1, 0, -1]
        )  # chat row, LoRA decision row, flagged base row
        native = lambda h: (
            h @ torch.randn(6, 12, generator=torch.Generator().manual_seed(1))
        )
        out = r.selective(hidden, sampler, native)
        before = out.clone()
        expected_heads = hidden @ r.weight.T
        torch.testing.assert_close(
            out[1, r.token_ids], expected_heads[1]
        )  # LoRA row: heads as before
        self.assertTrue(torch.isinf(out[1, [0, 1, 3]]).all())
        out = core.apply_base_heads(out, [2], core.LAST_STEP)
        torch.testing.assert_close(
            out[0], before[0]
        )  # unflagged base row (ordinary generation) untouched
        torch.testing.assert_close(out[1], before[1])
        torch.testing.assert_close(out[2, r.token_ids], expected_heads[2])
        mask = torch.ones(12, dtype=torch.bool)
        mask[r.token_ids] = False
        self.assertTrue(torch.isinf(out[2, mask]).all() and (out[2, mask] < 0).all())

    def test_without_flagged_rows_logits_are_identical(self):
        r = readout()
        hidden = torch.randn(2, 6)
        logits = r.selective(
            hidden, torch.tensor([-1, -1]), lambda h: h @ torch.ones(6, 12)
        )
        self.assertIs(core.apply_base_heads(logits, [], core.LAST_STEP), logits)

    def test_flagged_lora_row_is_left_unchanged(self):
        r = readout()
        logits = r.selective(
            torch.randn(2, 6), torch.tensor([-1, 0]), lambda h: h @ torch.ones(6, 12)
        )
        before = logits.clone()
        out = core.apply_base_heads(
            logits, [1], core.LAST_STEP
        )  # no exception, no device-side assert
        torch.testing.assert_close(out, before)

    def test_disabled_heads_or_row_mismatch_fail_loudly(self):
        core.LAST_STEP.set(None, None, None)
        r = readout(base_row_heads=False)
        logits = r.selective(
            torch.randn(2, 6), torch.tensor([-1, -1]), lambda h: h @ torch.ones(6, 12)
        )
        with self.assertRaises(RuntimeError):
            core.apply_base_heads(logits, [0], core.LAST_STEP)
        r = readout()
        r.selective(
            torch.randn(3, 6),
            torch.tensor([-1, -1, -1]),
            lambda h: h @ torch.ones(6, 12),
        )
        with self.assertRaises(RuntimeError):
            core.apply_base_heads(torch.zeros(2, 12), [0], core.LAST_STEP)

    def test_existing_behaviour_without_base_row_heads(self):
        a, b = readout(False), readout(True)
        hidden = torch.randn(3, 6)
        sampler = torch.tensor([-1, 0, 0])
        native = lambda h: h @ torch.ones(6, 12)
        torch.testing.assert_close(
            a.selective(hidden, sampler, native), b.selective(hidden, sampler, native)
        )


if __name__ == "__main__":
    unittest.main()


class ProcessorWithVllmTests(unittest.TestCase):
    def test_processor_with_real_vllm_classes(self):
        try:
            from vllm import SamplingParams
            from vllm.v1.sample.logits_processor.interface import (
                BatchUpdate,
                MoveDirectionality,
            )
        except Exception:
            self.skipTest("vLLM not installed")
        import json, os, tempfile
        from systemone_vllm_heads.processor import SystemOneBaseHeadsProcessor

        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "heads.json"
            manifest.write_text(json.dumps({"base_row_heads": False}))
            os.environ["SYSTEMONE_HEADS_MANIFEST"] = str(manifest)
            with self.assertRaises(ValueError):
                SystemOneBaseHeadsProcessor(None, torch.device("cpu"), False)
            manifest.write_text(json.dumps({"base_row_heads": True}))
            p = SystemOneBaseHeadsProcessor(None, torch.device("cpu"), False)
        flag = SamplingParams(max_tokens=1, extra_args={"systemone_head": "base"})
        plain = SamplingParams(max_tokens=1)
        p.update_state(
            BatchUpdate(
                batch_size=3,
                removed=[],
                added=[(0, plain, [1], []), (1, flag, [1], []), (2, flag, [1], [])],
                moved=[],
            )
        )
        p.update_state(
            BatchUpdate(
                batch_size=2,
                removed=[2],
                added=[],
                moved=[(1, 2, MoveDirectionality.SWAP)],
            )
        )
        self.assertEqual(sorted(p.flags), [2])
        r = readout()
        logits = r.selective(
            torch.randn(3, 6),
            torch.tensor([-1, -1, -1]),
            lambda h: torch.zeros(h.shape[0], 12),
        )
        out = p.apply(logits.clone())
        torch.testing.assert_close(out[0], logits[0])
        self.assertEqual(int(torch.isfinite(out[2]).sum()), 4)


class PackageFileTests(unittest.TestCase):
    def test_hf_cache_symlink_accepted_and_escapes_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            blob = root / "blobs" / "abc"
            blob.parent.mkdir()
            blob.write_bytes(b"w")
            snap = root / "snapshots" / "rev"
            snap.mkdir(parents=True)
            (snap / "heads.safetensors").symlink_to(Path("../../blobs/abc"))
            path = core.package_file(snap, "heads.safetensors")
            self.assertEqual(path.read_bytes(), b"w")
            for bad in ("../x", "/etc/passwd", "a/../../x", "", None, "a\\..\\x"):
                with self.assertRaises(ValueError, msg=bad):
                    core.package_file(snap, bad)
