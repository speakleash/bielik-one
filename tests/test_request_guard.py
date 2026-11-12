"""The frontend's request guard (body cap, x-request-id) and the refusal of text that is not Unicode."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from bielik_one.api import remote_vllm
from bielik_one.api.http_guard import REQUEST_ID, RequestGuard, guarded
from bielik_one.api.remote_vllm import RemoteVllmEngine
from bielik_one.api.server import build_app, question_to_row, state_text


async def echo(request: Request):
    body = await request.body()
    return JSONResponse({"size": len(body), "request_id": REQUEST_ID.get()})


async def boom(request: Request):
    raise RuntimeError("boom")


def guarded_echo(limit=100):
    app = Starlette(routes=[Route("/echo", echo, methods=["POST"]), Route("/boom", boom, methods=["POST"])])
    return TestClient(RequestGuard(app, limit), raise_server_exceptions=False)


class BodyCapTests(unittest.TestCase):
    def test_body_within_cap_reaches_the_app_unchanged(self):
        r = guarded_echo().post("/echo", content=b"x" * 100)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["size"], 100)

    def test_declared_length_over_cap_is_refused(self):
        r = guarded_echo().post("/echo", content=b"x" * 101)
        self.assertEqual(r.status_code, 413)
        self.assertIn("100 bytes", r.json()["error"])
        self.assertEqual(r.json()["code"], "PAYLOAD_TOO_LARGE")
        self.assertTrue(r.headers["x-request-id"].startswith("req-"))

    def test_chunked_body_over_cap_is_refused(self):
        def chunks():
            for _ in range(5):
                yield b"x" * 30

        r = guarded_echo().post("/echo", content=chunks())
        self.assertEqual(r.status_code, 413)

    def test_cap_from_megabytes(self):
        self.assertEqual(guarded(object(), 0.5).max_body_bytes, 2**19)
        with self.assertRaises(ValueError):
            RequestGuard(object(), 0)


class RequestIdTests(unittest.TestCase):
    def test_client_id_is_kept_and_visible_to_the_app(self):
        r = guarded_echo().post("/echo", content=b"{}", headers={"X-Request-Id": "abc-123"})
        self.assertEqual(r.headers["x-request-id"], "abc-123")
        self.assertEqual(r.json()["request_id"], "abc-123")

    def test_invalid_client_id_is_replaced(self):
        for bad in ("", "a b", "x" * 129, "ąę"):
            r = guarded_echo().post("/echo", content=b"{}", headers={"X-Request-Id": bad.encode()})
            self.assertTrue(r.headers["x-request-id"].startswith("req-"), bad)
            self.assertEqual(r.json()["request_id"], r.headers["x-request-id"])

    def test_generated_ids_differ(self):
        client = guarded_echo()
        ids = {client.post("/echo", content=b"{}").headers["x-request-id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_server_error_carries_the_id(self):
        r = guarded_echo().post("/boom", content=b"{}", headers={"X-Request-Id": "trace-me"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.headers["x-request-id"], "trace-me")

    def test_backend_reads_carry_the_request_id(self):
        seen = []

        def handler(request):
            seen.append(request.headers.get("x-request-id"))
            ids = json.loads(request.content)["logprob_token_ids"]
            return httpx.Response(200, json={
                "usage": {"prompt_tokens": 2},
                "choices": [{"logprobs": {"top_logprobs": [{f"token_id:{i}": -1.0 for i in ids}]}}],
            })

        engine = RemoteVllmEngine(
            dict(choice_capacity=255, labels_capacity=2, label_ids=[1000, 1001], anchor_id=1500,
                 bias=[0.0, 0.0], threshold=0.5),
            "http://backend", tokenizer=object(), client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(engine.close)
        rows = [SimpleNamespace(id=f"q{i}", question=SimpleNamespace(type="choice"), options=[0, 1]) for i in range(3)]
        token = REQUEST_ID.set("req-abc")
        try:
            with patch.object(remote_vllm, "inference_ids", lambda *a, **k: [10, 20]), \
                    patch.object(remote_vllm, "answer_token_ids", lambda *a, **k: [2000, 2001]):
                engine.distributions(rows)
        finally:
            REQUEST_ID.reset(token)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(s.startswith("req-abc-") for s in seen), seen)
        self.assertEqual(len(set(seen)), 3)  # one id per backend call

        seen.clear()
        engine._read([10, 20], [2000, 2001], "choice")  # outside a request: no header
        self.assertEqual(seen, [None])


class Engine:
    """Renders every question and encodes the text as a tokenizer would."""

    def evaluate(self, body):
        state = state_text(body.state)
        for name, question in body.questions.items():
            row, _ = question_to_row(state, name, question)
            (row.state + row.question.instructions).encode("utf-8")
        return {"model": "x", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


class LoneSurrogateTests(unittest.TestCase):
    def post(self, raw):
        client = TestClient(build_app(Engine()), raise_server_exceptions=False)
        return client.post("/v1/systemone", content=raw.encode("utf-8", "surrogatepass"),
                           headers={"content-type": "application/json"})

    def test_lone_surrogates_are_a_request_error(self):
        q = '"questions": {"q": {"type": "noul", "instructions": "ok?"}}'
        cases = {
            "state": '{"model": "m", "state": "\\udead", %s}' % q,
            "json state": '{"model": "m", "state": {"a": [1, {"b": "\\udfff"}]}, %s}' % q,
            "instructions": '{"model": "m", "state": "x", "questions": {"q": {"type": "noul", "instructions": "\\ud800"}}}',
            "criteria key": '{"model": "m", "state": "x", "questions": {"q": {"type": "choice", "instructions": "?", '
                            '"criteria": {"\\udead": "\\udead", "b": "y"}}}}',
            "question name": '{"model": "m", "state": "x", "questions": {"\\udead": {"type": "noul", "instructions": "?"}}}',
            "model": '{"model": "\\udead", "state": "x", %s}' % q,
        }
        for name, raw in cases.items():
            r = self.post(raw)
            self.assertEqual(r.status_code, 422, (name, r.text))
            self.assertEqual(r.json()["detail"][0]["loc"][0], "body", name)

    def test_message_names_the_place(self):
        q = '"questions": {"q": {"type": "noul", "instructions": "ok?"}}'
        r = self.post('{"model": "m", "state": {"a": [1, {"b": "\\udfff"}]}, %s}' % q)
        self.assertIn("state.a.1.b", r.json()["detail"][0]["msg"])

    def test_surrogate_pairs_are_ordinary_text(self):
        q = '"questions": {"q": {"type": "noul", "instructions": "ok?"}}'
        self.assertEqual(self.post('{"model": "m", "state": "\\ud83d\\ude00", %s}' % q).status_code, 200)


if __name__ == "__main__":
    unittest.main()
