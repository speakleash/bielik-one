"""Per-request resource limits of the remote frontend: total prompt tokens, queued reads after a failure, deadline."""

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from bielik_one import launch
from bielik_one.api import remote_vllm
from bielik_one.api.remote_vllm import BackendReadError, RemoteVllmEngine
from bielik_one.api.server import RequestError, SystemOneIn

from tests.test_launch import dry, package

METADATA = dict(
    choice_capacity=255,
    labels_capacity=248,
    label_ids=list(range(1000, 1248)),
    anchor_id=1500,
    bias=[0.0] * 248,
    threshold=0.5,
    source_repo="test/decision",
)


def answer(request):
    ids = json.loads(request.content)["logprob_token_ids"]
    values = {f"token_id:{i}": -1.0 for i in ids}
    return httpx.Response(
        200,
        json={
            "usage": {"prompt_tokens": len(json.loads(request.content)["prompt"])},
            "choices": [{"logprobs": {"top_logprobs": [values]}}],
        },
    )


def body(questions=8, permutations=16, options=4):
    """`questions` choice questions with `options` options each: questions x permutations reads."""
    return SystemOneIn(
        model="decision",
        state="x",
        permutations=permutations,
        read_twice=False,
        questions={
            f"q{i}": {
                "type": "choice",
                "instructions": "?",
                "criteria": {f"o{j}": "t" for j in range(options)},
            }
            for i in range(questions)
        },
    )


@contextlib.contextmanager
def prompts(length=50, counter=None):
    """Every prompt is `length` tokens; `counter` (a list) records each tokenized prompt."""

    def tokenize(*args, **kwargs):
        if counter is not None:
            counter.append(1)
        return [7] * length

    letters = lambda tok, row, semantic: list(range(1000, 1000 + len(row.options)))
    with patch("bielik_one.api.remote_vllm.inference_ids", side_effect=tokenize), patch(
        "bielik_one.api.remote_vllm.answer_token_ids", side_effect=letters
    ), patch("bielik_one.api.server.answer_token_ids", side_effect=letters):
        yield


class Backend:
    """A mock vLLM that counts reads; `handle` decides each answer from its 1-based number."""

    def __init__(self, handle=None):
        self.calls = 0
        self.lock = threading.Lock()
        self.handle = handle

    def __call__(self, request):
        with self.lock:
            self.calls += 1
            number = self.calls
        return self.handle(number, request) if self.handle else answer(request)


def engine(backend, workers=2, **limits):
    return RemoteVllmEngine(
        METADATA,
        "http://backend",
        tokenizer=object(),
        workers=workers,
        client=httpx.Client(transport=httpx.MockTransport(backend)),
        **limits,
    )


class TokenLimitTests(unittest.TestCase):
    def test_request_within_the_limit_is_answered(self):
        backend = Backend()
        e = engine(backend, max_request_tokens=128 * 50)
        self.addCleanup(e.close)
        with prompts(50):
            result = e.evaluate(body(questions=8, permutations=16))
        self.assertEqual(len(result["answers"]), 8)
        self.assertEqual(backend.calls, 128)

    def test_oversized_request_is_refused_before_any_read(self):
        backend = Backend()
        tokenized = []
        e = engine(backend, max_request_tokens=128 * 50 - 1)
        self.addCleanup(e.close)
        with prompts(50, tokenized), self.assertRaises(RequestError) as caught:
            e.evaluate(body(questions=8, permutations=16))
        self.assertEqual(caught.exception.loc, ["body", "questions"])
        self.assertIn(str(128 * 50 - 1), caught.exception.msg)
        self.assertEqual(backend.calls, 0)
        # It stops at the prompt that crosses the limit instead of building every prompt first.
        self.assertEqual(len(tokenized), 128)

    def test_refusal_stops_early_for_a_much_larger_request(self):
        tokenized = []
        e = engine(Backend(), max_request_tokens=1000)
        self.addCleanup(e.close)
        with prompts(400, tokenized), self.assertRaises(RequestError):
            e.evaluate(body(questions=8, permutations=16))
        self.assertEqual(len(tokenized), 3)

    def test_limit_can_be_lifted(self):
        backend = Backend()
        e = engine(backend, max_request_tokens=None)
        self.addCleanup(e.close)
        with prompts(50):
            e.evaluate(body(questions=8, permutations=16))
        self.assertEqual(backend.calls, 128)

    def test_invalid_limits_are_rejected(self):
        for limits in ({"max_request_tokens": 0}, {"request_timeout": 0}, {"request_timeout": float("inf")}):
            with self.assertRaises(ValueError, msg=limits):
                engine(Backend(), **limits)

    def test_http_status_is_422(self):
        e = engine(Backend(), max_request_tokens=10)
        self.addCleanup(e.close)
        client = TestClient(remote_vllm.build_remote_app(e, chat_model="chat"))
        with prompts(50):
            response = client.post("/v1/systemone", json=body(1, 1).model_dump())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INVALID_REQUEST")


class QueuedReadsTests(unittest.TestCase):
    def test_reads_still_queued_are_dropped_after_a_failure(self):
        release = threading.Event()

        def handle(number, request):
            if number == 1:
                return httpx.Response(500)
            release.wait(5)  # keep the other worker busy until the request has given up
            return answer(request)

        backend = Backend(handle)
        e = engine(backend, workers=2)
        self.addCleanup(e.close)
        try:
            with prompts(50), self.assertRaises(BackendReadError):
                e.evaluate(body(questions=8, permutations=16))
        finally:
            release.set()
        time.sleep(0.3)  # the pool keeps running between requests: give it time to pick up anything left
        # The failed read, the one running beside it, and at most one taken by the freed worker before cancellation.
        self.assertLessEqual(backend.calls, 3)


class DeadlineTests(unittest.TestCase):
    def slow(self, delay):
        def handle(number, request):
            time.sleep(delay)
            return answer(request)

        return Backend(handle)

    def test_request_over_its_deadline_fails_with_a_timeout(self):
        backend = self.slow(0.1)
        e = engine(backend, workers=2, request_timeout=0.5)
        self.addCleanup(e.close)
        started = time.monotonic()
        with prompts(50), self.assertRaises(BackendReadError) as caught:
            e.evaluate(body(questions=8, permutations=16))  # 128 reads x 0.1 s / 2 workers = 6.4 s unbounded
        held = time.monotonic() - started
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertLess(held, 1.5)
        time.sleep(0.3)
        self.assertLess(backend.calls, 30)

    def test_request_within_its_deadline_is_answered(self):
        e = engine(self.slow(0.01), workers=2, request_timeout=30)
        self.addCleanup(e.close)
        with prompts(50):
            self.assertEqual(len(e.evaluate(body(questions=2, permutations=4))["answers"]), 2)

    def test_http_status_is_504(self):
        e = engine(self.slow(0.2), workers=1, request_timeout=0.1)
        self.addCleanup(e.close)
        client = TestClient(remote_vllm.build_remote_app(e, chat_model="chat"))
        with prompts(50):
            response = client.post("/v1/systemone", json=body(2, 4).model_dump())
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["code"], "TIMEOUT")


class LauncherTests(unittest.TestCase):
    def test_limits_go_to_the_frontend(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                command = dry("frontend", d)["command"]
                self.assertNotIn("--max-request-tokens", command)
                self.assertNotIn("--request-timeout", command)
                command = dry("frontend", d, "--max-request-tokens", "1000", "--request-timeout", "30")["command"]
                self.assertEqual(command[command.index("--max-request-tokens") + 1], "1000", kind)
                self.assertEqual(command[command.index("--request-timeout") + 1], "30.0", kind)

    def test_invalid_limits_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            for args in (
                ["--max-request-tokens", "0"],
                ["--request-timeout", "0"],
                ["--request-timeout", "inf"],
                ["--request-timeout", "nan"],
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=args):
                    launch.main(["frontend", d, *args, "--dry-run"])

    def test_frontend_parsers_accept_the_limits(self):
        import argparse

        parser = argparse.ArgumentParser()
        remote_vllm.add_limit_arguments(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.max_request_tokens, remote_vllm.DEFAULT_MAX_REQUEST_TOKENS)
        self.assertEqual(defaults.request_timeout, remote_vllm.DEFAULT_REQUEST_TIMEOUT)
        parsed = parser.parse_args(["--max-request-tokens", "1000", "--request-timeout", "30.0"])
        self.assertEqual((parsed.max_request_tokens, parsed.request_timeout), (1000, 30.0))
        for args in (["--max-request-tokens", "0"], ["--request-timeout", "-1"], ["--request-timeout", "inf"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=args):
                parser.parse_args(args)


if __name__ == "__main__":
    unittest.main()
