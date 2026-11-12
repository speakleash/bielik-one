"""One error shape with a machine code on every route; the OpenAI shape on the chat routes; the frontends as wired."""

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from bielik_one import launch
from bielik_one.api import remote_vllm
from bielik_one.api.http_guard import REQUEST_ID, RequestGuard, guarded, max_body_mb
from bielik_one.api.merged_vllm import MaskChatTokens
from bielik_one.api.reasoning import EndpointReasoner, ReasoningError
from bielik_one.api.remote_vllm import BackendReadError
from bielik_one.api.server import RequestError, build_app, systemone_answer

from tests.test_launch import dry, package

OK = {"model": "decision", "state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}


class Engine:
    """A ready engine for the routes: `fail` picks how /v1/systemone fails."""

    backend_url = "http://backend"
    aliases = {"decision"}
    temperatures = {}
    calibration_source = None

    def __init__(self, fail=None):
        self.fail = fail
        self.seen_ids = []

    def evaluate(self, body):
        self.seen_ids.append(REQUEST_ID.get())
        if self.fail == "request":
            raise RequestError(["questions", "q", "criteria"], "bad criteria")
        if self.fail == "backend":
            raise BackendReadError("vLLM did not return a complete decision read") from ValueError("x")
        if self.fail == "timeout":
            raise BackendReadError("vLLM did not return a complete decision read") from httpx.ReadTimeout("slow")
        if self.fail == "crash":
            raise RuntimeError("boom")
        if isinstance(self.fail, ReasoningError):
            raise self.fail
        return {"model": "decision", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def close(self):
        pass


def client(fail=None, fmt="default", limit=1000):
    engine = Engine(fail)
    app = guarded(remote_vllm.build_remote_app(engine, chat_model="chat", error_format=fmt), limit / 2**20, fmt)
    return TestClient(app, raise_server_exceptions=False), engine


class AnswerShapeTests(unittest.TestCase):
    def test_every_type_has_probabilities_and_confidence(self):
        answers = [
            systemone_answer("noul", ["no", "yes"], [0.3, 0.7]),
            systemone_answer("choice", ["a", "b"], [0.4, 0.6]),
            systemone_answer("score", ["0", "1"], [0.2, 0.8], ["low", "high"]),
            systemone_answer("labels", ["a", "b"], [0.9, 0.1]),
        ]
        for answer in answers:
            self.assertIn("probabilities", answer, answer["type"])
            self.assertIn("confidence", answer, answer["type"])
        noul = answers[0]
        self.assertEqual(noul["probabilities"], {"true": 0.7, "false": 0.3})
        self.assertEqual(noul["noul"], 0.7)


class DefaultFormatTests(unittest.TestCase):
    def test_validation_error_keeps_the_field_and_drops_the_input(self):
        c, _ = client()
        r = c.post("/v1/systemone", json={**OK, "state": "s" * 500, "questions": {"q": {"type": "box"}}})
        self.assertEqual(r.status_code, 422)
        body = r.json()
        self.assertEqual(body["code"], "INVALID_REQUEST")
        self.assertEqual(body["detail"][0]["loc"], ["body", "questions", "q", "type"])
        self.assertEqual(body["error"], body["detail"][0]["msg"])
        self.assertNotIn("s" * 500, r.text)

    def test_engine_request_error(self):
        r = client("request")[0].post("/v1/systemone", json=OK)
        self.assertEqual((r.status_code, r.json()["code"]), (422, "INVALID_REQUEST"))
        self.assertEqual(r.json()["detail"][0]["loc"], ["body", "questions", "q", "criteria"])

    def test_unparsable_body_is_invalid_request(self):
        r = client()[0].post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
        self.assertEqual((r.status_code, r.json()["code"]), (422, "INVALID_REQUEST"))

    def test_backend_failures(self):
        for fail, status, code in (("backend", 502, "BACKEND_UNAVAILABLE"), ("timeout", 504, "TIMEOUT")):
            r = client(fail)[0].post("/v1/systemone", json=OK)
            self.assertEqual((r.status_code, r.json()["code"]), (status, code), fail)

    def test_crash_is_json_with_the_request_id(self):
        r = client("crash")[0].post("/v1/systemone", json=OK, headers={"X-Request-Id": "crash-1"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json(), {"error": "internal server error", "code": "INTERNAL_ERROR"})
        self.assertEqual(r.headers["x-request-id"], "crash-1")

    def test_unknown_path_and_method(self):
        c, _ = client()
        self.assertEqual(c.get("/nope").json()["code"], "NOT_FOUND")
        r = c.get("/v1/systemone")
        self.assertEqual((r.status_code, r.json()["code"]), (405, "METHOD_NOT_ALLOWED"))

    def test_too_large_with_client_id(self):
        r = client()[0].post("/v1/systemone", content=b"x" * 2000, headers={"X-Request-Id": "big-1"})
        self.assertEqual((r.status_code, r.json()["code"]), (413, "PAYLOAD_TOO_LARGE"))
        self.assertEqual(r.headers["x-request-id"], "big-1")

    def test_backend_down(self):
        c, _ = client()
        with c:
            r = c.get("/health")
            self.assertEqual((r.status_code, r.json()["code"], r.json()["status"]),
                             (503, "BACKEND_UNAVAILABLE", "backend_unavailable"))
            r = c.get("/v1/models")
            self.assertEqual((r.status_code, r.json()["code"]), (502, "BACKEND_UNAVAILABLE"))


    def test_reasoning_failures(self):
        for status, http, code in ((422, 422, "INVALID_REQUEST"), (504, 504, "TIMEOUT"), (502, 502, "BACKEND_UNAVAILABLE")):
            r = client(ReasoningError("upstream said: secret body", status))[0].post("/v1/systemone", json=OK)
            self.assertEqual((r.status_code, r.json()["code"]), (http, code), status)
        self.assertNotIn("secret", r.text)  # an upstream body is logged, not answered

    def test_reasoning_status_from_the_endpoint(self):
        import urllib.error
        from unittest.mock import MagicMock

        row = MagicMock()
        cases = {
            urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"")): 502,
            urllib.error.HTTPError("u", 429, "busy", {}, io.BytesIO(b"")): 502,
            urllib.error.HTTPError("u", 413, "long", {}, io.BytesIO(b"")): 422,
            urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"")): 422,
            urllib.error.URLError(TimeoutError("timed out")): 504,
            TimeoutError("timed out"): 504,
            urllib.error.URLError(ConnectionRefusedError()): 502,
        }
        for error, status in cases.items():
            with patch("bielik_one.api.reasoning.reasoning_messages", lambda *a: []), \
                    patch("bielik_one.api.reasoning._OPENER") as opener:
                opener.open.side_effect = error
                with self.assertRaises(ReasoningError) as caught:
                    EndpointReasoner("http://r/v1", "m")(row, 16)
            self.assertEqual(caught.exception.status, status, repr(error))

    def test_method_not_allowed_keeps_allow(self):
        r = client()[0].get("/v1/systemone")
        self.assertIn("POST", r.headers["allow"])


class OpenAIFormatTests(unittest.TestCase):
    def assert_openai(self, r, status, kind):
        error = r.json()["error"]
        self.assertEqual(r.status_code, status)
        self.assertEqual(set(error), {"message", "type", "param", "code"})
        self.assertEqual((error["type"], error["code"]), (kind, status))
        return error

    def test_chat_routes_answer_in_the_openai_format(self):
        c, _ = client()
        r = c.post("/v1/chat/completions", json={"model": "decision", "messages": []})
        self.assert_openai(r, 400, "invalid_request_error")
        r = c.post("/v1/completions", content=b"x" * 2000)
        self.assert_openai(r, 413, "invalid_request_error")

    def test_flag_switches_every_route(self):
        c, _ = client(fmt="openai")
        error = self.assert_openai(c.post("/v1/systemone", json={**OK, "questions": {"q": {"type": "box"}}}),
                                   422, "invalid_request_error")
        self.assertEqual(error["param"], "questions.q.type")
        self.assert_openai(client("backend", fmt="openai")[0].post("/v1/systemone", json=OK), 502, "server_error")

    def test_parser_offset_is_not_a_param(self):
        r = client(fmt="openai")[0].post("/v1/systemone", content=b"{not json",
                                        headers={"content-type": "application/json"})
        self.assertIsNone(self.assert_openai(r, 422, "invalid_request_error")["param"])

    def test_chat_backend_failures(self):
        for raised, status in ((httpx.ConnectError("down"), 502), (httpx.ReadTimeout("slow"), 504)):
            async def send(self, request, stream=False, raised=raised):
                raise raised

            with patch.object(httpx.AsyncClient, "send", send):
                r = client()[0].post("/v1/chat/completions", json={"messages": []})
            self.assert_openai(r, status, "server_error")

    def test_vllm_errors_pass_through_unchanged(self):
        vllm_error = {"error": {"message": "context too long", "type": "BadRequestError", "param": None, "code": 400}}

        async def body():
            yield json.dumps(vllm_error).encode()

        async def send(self, request, stream=False):
            return httpx.Response(400, headers={"content-type": "application/json"}, content=body())

        async def get(self, url, **kw):
            return httpx.Response(503, json={"object": "error", "message": "loading"}, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "send", send):
            r = client()[0].post("/v1/chat/completions", json={"messages": []})
        self.assertEqual((r.status_code, r.json()), (400, vllm_error))
        with patch.object(httpx.AsyncClient, "get", get):
            r = client()[0].get("/v1/models")
        self.assertEqual((r.status_code, r.json()["message"]), (503, "loading"))

    def test_masked_chat_errors(self):
        app = guarded(MaskChatTokens(remote_vllm.build_remote_app(Engine(), chat_model="chat"), ["<x>"]), 1)
        r = TestClient(app).post("/v1/chat/completions", content=b"[1]")
        self.assertEqual(r.json()["error"]["code"], 400)


class WiredFrontendTests(unittest.TestCase):
    def test_request_id_reaches_the_sync_endpoint(self):
        c, engine = client()
        with c:  # runs the lifespan through the guard
            r = c.post("/v1/systemone", json=OK, headers={"X-Request-Id": "abc"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(engine.seen_ids, ["abc"])
        self.assertEqual(r.headers["x-request-id"], "abc")

    def test_chat_proxy_streams_and_passes_the_id(self):
        sent = []

        async def chunks():
            for part in (b"data: 1\n\n", b"data: 2\n\n"):
                await asyncio.sleep(0)
                yield part

        async def send(self, request, stream=False):
            sent.append((request.headers.get("x-request-id"), json.loads(request.content)))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

        for masked in (False, True):
            sent.clear()
            app = remote_vllm.build_remote_app(Engine(), chat_model="chat")
            if masked:
                app = MaskChatTokens(app, ["<x>"])
            with patch.object(httpx.AsyncClient, "send", send), TestClient(guarded(app, 1)) as c:
                r = c.post("/v1/chat/completions", json={"messages": [], "stream": True},
                           headers={"X-Request-Id": "chat-7"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.text, "data: 1\n\ndata: 2\n\n")
            self.assertEqual(r.headers["x-request-id"], "chat-7")
            backend_id, body = sent[0]
            self.assertTrue(backend_id.startswith("chat-7-"), backend_id)
            self.assertEqual(body["model"], "chat")
            self.assertEqual("bad_words" in body, masked)


class DisconnectTests(unittest.TestCase):
    def test_client_gone_while_sending_its_body(self):
        called, sent = [], []

        async def app(scope, receive, send):
            called.append(scope)

        async def run():
            messages = iter([{"type": "http.request", "body": b"x", "more_body": True}, {"type": "http.disconnect"}])

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            await RequestGuard(app, 100)({"type": "http", "method": "POST", "path": "/v1/systemone",
                                          "headers": []}, receive, send)

        asyncio.run(run())
        self.assertEqual((called, sent), ([], []))


class FlagTests(unittest.TestCase):
    def test_max_body_mb_type(self):
        import argparse

        self.assertEqual(max_body_mb("0.5"), 0.5)
        for bad in ("0", "-1", "nan", "inf", "1e-9", "x"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                max_body_mb(bad)

    def test_launcher_passes_and_checks_the_flags(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            front = dry("frontend", d, "--error-format", "openai")["command"]
            self.assertEqual(front[front.index("--error-format") + 1], "openai")
            self.assertNotIn("--error-format", dry("frontend", d)["command"])
            for args in (["--max-body-mb", "inf"], ["--max-body-mb", "1e-9"], ["--error-format", "xml"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launch.main(["frontend", d, *args, "--dry-run"])


if __name__ == "__main__":
    unittest.main()
