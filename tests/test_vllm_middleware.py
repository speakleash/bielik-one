"""Exercise the middleware in vLLM's FastAPI/auth/CORS layout, without GPU weights."""

from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from bielik_one.api import vllm_middleware as middleware
from bielik_one.api.server import RequestError, systemone_answer


class Engine:
    backend_url = "http://unbound"
    temperatures = {"1": {"choice": 1.5}}
    calibration_source = "calibration.json"

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        self.aliases = set()
        self.reasoners = {}
        self.model_name = "unbound"
        self.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.closed = 0
        self.requests = []

    def close(self):
        self.closed += 1
        self.pool.shutdown(wait=True)
        self.client.close()

    def evaluate(self, body):
        if body.model not in self.aliases:
            raise RequestError(["model"], "unknown model")
        self.requests.append(body)
        answers = {}
        for name, question in body.questions.items():
            keys = ["no", "yes"] if question.type == "noul" else list(question.criteria)
            answers[name] = systemone_answer(question.type, keys, [0.2, 0.8])
        return {"model": body.model, "answers": answers, "usage": {"input_tokens": 3, "output_tokens": 1}}


class ServerAuth:
    """vLLM authenticates before routing and lets CORS handle OPTIONS."""

    def __init__(self, app, keys):
        self.app, self.keys = app, keys

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] != "OPTIONS" and self.keys:
            header = middleware._header(scope.get("headers", []), b"authorization") or ""
            scheme, _, key = header.partition(" ")
            if scheme.lower() != "bearer" or key not in self.keys:
                return await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
        return await self.app(scope, receive, send)


def make_app(*, keys=(), env_key=None, limit=16, cards=None, root_path="", fail_startup=False):
    engine = Engine()
    startup = []

    @asynccontextmanager
    async def lifespan(app):
        startup.append("startup")
        if fail_startup:
            raise RuntimeError("vLLM startup failed")
        try:
            yield {"preserved_state": True}
        finally:
            startup.append("shutdown")

    app = FastAPI(lifespan=lifespan, root_path=root_path)
    app.state.args = SimpleNamespace(api_key=list(keys))
    cards = cards if cards is not None else [
        {"id": "base-a", "parent": None},
        {"id": "base-b", "parent": None},
        {"id": "systemone-decision", "parent": "base-a"},
    ]

    @app.get("/v1/models")
    def models():
        return {"data": cards}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    @app.post("/v1/responses")
    async def generate(request: Request):
        body = await request.json()
        if body.get("stream"):
            async def chunks():
                yield b"data: one\n\n"
                yield b"data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        return {"received": body}

    @app.post("/upload")
    async def upload(request: Request):
        return {"bytes": len(await request.body())}

    app.add_middleware(CORSMiddleware, allow_origins=["https://client.example"],
                       allow_methods=["POST"], allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(ServerAuth, keys=keys or ([env_key] if env_key else []))
    app.add_middleware(middleware.SystemOneMiddleware,
                       settings=middleware.Settings(api_key=env_key, max_body_mb=limit), engine=engine)
    app.middleware_stack = app.build_middleware_stack()
    owner = app.middleware_stack
    while not isinstance(owner, middleware.SystemOneMiddleware):
        owner = owner.app
    owner.check_transport = httpx.ASGITransport(app=app)
    return app, owner, engine, startup


def decision(model="base-a", kind="choice"):
    criteria = ["low", "high"] if kind == "score" else {"a": "First", "b": "Second"}
    return {"model": model, "state": "Context", "questions": {
        "q": {"type": kind, "instructions": "Question?", "criteria": criteria}
    }}


@pytest.mark.parametrize("content_type", [None, "application/json", "Application/JSON", "application/problem+json"])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/responses"])
def test_internal_adapter_is_never_public(content_type, path):
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        headers = {"content-type": content_type} if content_type else {}
        response = client.post(path, content=json.dumps({"model": "systemone-decision"}).encode(), headers=headers)
        assert response.status_code == 400


@pytest.mark.parametrize("content_type", [None, "Application/JSON", "application/problem+json"])
def test_untrusted_backend_flags_removed(content_type):
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        headers = {"content-type": content_type} if content_type else {}
        response = client.post("/v1/completions", content=json.dumps({
            "model": "base-a", "prompt": "hello", "vllm_xargs": {"systemone": True}
        }).encode(), headers=headers)
        assert response.status_code == 200
        assert "vllm_xargs" not in response.json()["received"]
        assert client.post("/v1/completions", content=b'{"prompt_logprobs":1}', headers=headers).status_code == 400


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/chat/completions"])
def test_cli_or_config_keys_protect_both_routes(path):
    app, owner, engine, _ = make_app(keys=["config-key", "second-key"], env_key="overridden-env-key")
    with TestClient(app) as client:
        body = decision() if path == "/v1/systemone" else {"model": "base-a"}
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"Authorization": "Bearer overridden-env-key"}).status_code == 401
        for key in ("config-key", "second-key"):
            assert client.post(path, json=body, headers={"Authorization": f"Bearer {key}"}).status_code == 200
        assert engine.client.headers["Authorization"] == "Bearer config-key"
        if path == "/v1/systemone":
            assert owner.engine.reasoners["original"].api_key == "config-key"


def test_env_key_is_used_when_no_cli_key_is_set():
    app, _, engine, _ = make_app(env_key="env-key")
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=decision()).status_code == 401
        assert client.post("/v1/systemone", json=decision(), headers={"Authorization": "Bearer env-key"}).status_code == 200
        assert engine.client.headers["Authorization"] == "Bearer env-key"


def test_cors_preflight_success_and_errors_use_vllm_configuration():
    app, _, _, _ = make_app(keys=["key"])
    with TestClient(app) as client:
        response = client.options("/v1/systemone", headers={
            "Origin": "https://client.example", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://client.example"
        headers = {"Origin": "https://client.example", "Authorization": "Bearer key"}
        for path, body in [("/v1/systemone", decision()),
                           ("/v1/chat/completions", {"model": "systemone-decision"})]:
            response = client.post(path, json=body, headers=headers)
            assert response.status_code in (200, 400)
            assert response.headers["access-control-allow-origin"] == "https://client.example"


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/completions"])
def test_bodies_are_limited_before_json_parsing(path):
    app, _, engine, _ = make_app(limit=100 / 2**20)
    with TestClient(app) as client:
        response = client.post(path, content=b"x" * 101, headers={"content-type": "application/json"})
        assert response.status_code == 413
        assert not engine.requests
        response = client.post(path, content=iter([b"x" * 60, b"x" * 60]), headers={"content-type": "application/json"})
        assert response.status_code == 413


def test_non_json_routes_and_streaming_still_work():
    app, _, _, _ = make_app(limit=100 / 2**20)
    with TestClient(app) as client:
        assert client.post("/upload", content=b"x" * 150, headers={"content-type": "multipart/form-data"}).json() == {"bytes": 150}
        response = client.post("/v1/chat/completions", json={"model": "base-a", "stream": True})
        assert response.status_code == 200
        assert response.content == b"data: one\n\ndata: [DONE]\n\n"


@pytest.mark.parametrize("kind", ["choice", "noul", "score", "labels"])
def test_decisions_reuse_engine_and_public_aliases(kind):
    app, owner, engine, _ = make_app()
    with TestClient(app) as client:
        for model in ("base-a", "base-b"):
            response = client.post("/v1/systemone", json=decision(model, kind))
            assert response.status_code == 200
            answer = response.json()["answers"]["q"]
            assert answer["type"] == kind and "probabilities" in answer
        assert engine.aliases == {"base-a", "base-b"}
        assert owner.public_names == ["base-a", "base-b"]
        assert client.post("/v1/systemone", json=decision("decision")).status_code == 422
        assert client.get("/v1/systemone").status_code == 405
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["base-a", "base-b"]
        # Internal reads need the adapter, but a wrong token must never reveal it.
        assert len(client.get("/v1/models", headers={"x-systemone-internal": "wrong"}).json()["data"]) == 2
        assert len(client.get("/v1/models", headers={"x-systemone-internal": owner.settings.internal_token}).json()["data"]) == 3


@pytest.mark.parametrize("cards", [
    [{"id": "base-a", "parent": None}],
    [{"id": "base-a", "parent": None}, {"id": "systemone-decision", "parent": None}],
])
def test_missing_or_misidentified_adapter_fails_before_inference(cards):
    app, _, engine, _ = make_app(cards=cards)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=decision())
        assert response.status_code == 503
        assert not engine.requests


def test_root_path_and_lifespan_cleanup():
    app, _, engine, events = make_app(root_path="/prefix")
    with TestClient(app) as client:
        assert not engine.closed
        response = client.post("/prefix/v1/systemone", json=decision())
        assert response.status_code == 200
    assert events == ["startup", "shutdown"]
    assert engine.closed == 1 and engine.client.is_closed and engine.pool._shutdown


def test_engine_is_closed_even_when_vllm_startup_fails():
    app, _, engine, _ = make_app(fail_startup=True)
    with pytest.raises(RuntimeError, match="vLLM startup failed"):
        with TestClient(app):
            pass
    assert engine.closed == 1 and engine.client.is_closed and engine.pool._shutdown


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_invalid_body_limits_are_rejected(value):
    with pytest.raises(RuntimeError, match="SYSTEMONE_MAX_BODY_MB"):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MAX_BODY_MB": value})


def test_existing_decision_route_is_not_replaced_or_loaded():
    app = FastAPI()
    app.add_api_route("/v1/systemone", lambda: {"native": True}, methods=["POST"])
    with patch.object(middleware, "build_engine") as build:
        with pytest.raises(RuntimeError, match="already defines"):
            middleware.SystemOneMiddleware(app, settings=middleware.Settings())
        build.assert_not_called()


@pytest.mark.parametrize("kind", ["llama-merged", "qwen-merged"])
def test_merged_packages_are_rejected_explicitly(kind, tmp_path):
    from tests.test_launch import package

    package(kind, tmp_path)
    with pytest.raises(RuntimeError, match="serves adapter packages"):
        middleware.build_engine(middleware.Settings(package=tmp_path))


def test_build_engine_reuses_adapter_metadata_and_calibration(tmp_path, monkeypatch):
    from tests.test_launch import package
    from bielik_one.api import remote_vllm

    package("llama-adapter", tmp_path)
    monkeypatch.setenv("SYSTEMONE_HEADS_MANIFEST", str(tmp_path / "heads.json"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    seen = {}

    def create(metadata, url, **kwargs):
        seen.update(metadata=metadata, url=url, **kwargs)
        return SimpleNamespace(aliases=set(), client=kwargs["client"])

    with patch.object(remote_vllm, "RemoteVllmEngine", side_effect=create), patch.object(
        remote_vllm, "resolve_calibration", return_value=({"1": {"choice": 2.0}}, "calibration.json")
    ):
        engine = middleware.build_engine(middleware.Settings(package=tmp_path))
    try:
        assert seen["metadata"]["adapter"] == str(tmp_path / "adapter")
        assert seen["metadata"]["heads_manifest"] == str(tmp_path / "heads.json")
        assert seen["backend_model"] == "systemone-decision"
        assert seen["calibration"] == {"1": {"choice": 2.0}}
        assert engine.aliases == set()
    finally:
        engine.client.close()


def test_explicit_reasoning_key_takes_precedence_over_environment(monkeypatch):
    from bielik_one.api import reasoning
    from bielik_one.prompting.schema import Option, Question, Row

    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    row = Row(id="q", source="test", group_id="test", state="Context", split="test", lang="pl", license="test",
              question=Question(type="choice", instructions="Question", criteria={"a": "A", "b": "B"}),
              options=[Option(key="a", text="A"), Option(key="b", text="B")], target_probs=[1.0, 0.0])
    reasoner = reasoning.EndpointReasoner("http://server/v1", "base-a", key_env="VLLM_API_KEY", api_key="config-key")
    response = SimpleNamespace(read=lambda: b'{"choices":[{"message":{"content":"analysis"}}]}')
    with patch.object(reasoning._OPENER, "open") as open_request:
        open_request.return_value.__enter__.return_value = response
        reasoner(row, 10)
        assert open_request.call_args.args[0].get_header("Authorization") == "Bearer config-key"
