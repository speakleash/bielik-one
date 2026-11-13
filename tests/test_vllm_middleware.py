"""Exercise the middleware in vLLM's FastAPI/auth/CORS layout, without GPU weights."""

from contextlib import asynccontextmanager
import json
from pathlib import Path
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


def make_app(*, keys=(), env_key=None, limit=16, cards=None, root_path="", fail_startup=False, engine=None,
             server_count=None, native_route=False, **settings):
    engine = engine if engine is not None else Engine()
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
    app.state.args = SimpleNamespace(api_key=list(keys), api_server_count=server_count)
    cards = cards if cards is not None else [
        {"id": "base-a", "parent": None},
        {"id": "base-b", "parent": None},
        {"id": "decision", "parent": "base-a"},
    ]

    @app.get("/v1/models")
    def models():
        return {"data": cards}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    @app.post("/v1/responses")
    @app.post("/v1/messages")
    @app.post("/inference/v1/generate")
    @app.post("/v1/load_lora_adapter")
    @app.post("/v1/unload_lora_adapter")
    async def generate(request: Request):
        body = await request.json()
        if body.get("stream"):
            async def chunks():
                yield b"data: one\n\n"
                yield b"data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        return {"received": body}

    @app.post("/adapters")
    @app.delete("/adapters/{name}")
    async def adapters():
        return {"ok": True}

    @app.post("/invocations")
    async def invocations(request: Request):
        # vLLM's SageMaker route copies the adapter header into the model after any body filter.
        body = await request.json()
        body["model"] = request.headers.get("x-amzn-sagemaker-adapter-identifier", body.get("model"))
        return {"received": body}

    @app.post("/upload")
    async def upload(request: Request):
        return {"bytes": len(await request.body())}

    if native_route:
        async def create_structured_decision():
            return JSONResponse({"error": "The model does not support the structured decision API"}, 501)

        create_structured_decision.__module__ = "vllm.entrypoints.generate.structured_decisions.api_router"
        app.add_api_route("/v1/systemone", create_structured_decision, methods=["POST"])

    app.add_middleware(CORSMiddleware, allow_origins=["https://client.example"],
                       allow_methods=["POST"], allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(ServerAuth, keys=keys or ([env_key] if env_key else []))
    app.add_middleware(middleware.SystemOneMiddleware,
                       settings=middleware.Settings(api_key=env_key, max_body_mb=limit, **settings), engine=engine)
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
        response = client.post(path, content=json.dumps({"model": "decision"}).encode(), headers=headers)
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
                           ("/v1/chat/completions", {"model": "decision"})]:
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
    [{"id": "base-a", "parent": None}, {"id": "decision", "parent": None}],
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
        assert seen["backend_model"] == "decision"
        assert seen["workers"] == 8  # `bielik-one frontend` value for Llama adapter packages
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


class CountingTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def handle_async_request(self, request):
        self.calls += 1
        return await self.inner.handle_async_request(request)


def test_multiple_api_servers_require_a_shared_internal_token():
    app, _, engine, events = make_app(server_count=2)
    with pytest.raises(RuntimeError, match="SYSTEMONE_INTERNAL_TOKEN"):
        with TestClient(app):
            pass
    assert engine.closed == 1 and not events  # vLLM startup is not entered
    app, _, _, _ = make_app(server_count=2, internal_token="shared", shared_token=True)
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=decision()).status_code == 200


def test_settings_from_env():
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package"})
    assert settings.lora_name == "decision" and settings.workers is None and not settings.shared_token
    settings = middleware.Settings.from_env(
        {"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_INTERNAL_TOKEN": "s", "SYSTEMONE_WORKERS": "32"}
    )
    assert settings.internal_token == "s" and settings.shared_token and settings.workers == 32
    with pytest.raises(RuntimeError, match="SYSTEMONE_WORKERS"):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_WORKERS": "many"})


def test_non_json_content_type_is_still_filtered():
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        headers = {"content-type": "text/plain"}
        response = client.post("/v1/messages", content=b'{"model": "decision"}', headers=headers)
        assert response.status_code == 400
        response = client.post("/v1/messages", content=b'{"model": "base-a", "vllm_xargs": {"a": 1}}', headers=headers)
        assert response.status_code == 200 and "vllm_xargs" not in response.json()["received"]


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/v1/load_lora_adapter", {"lora_name": "copy", "lora_path": "/pkg/adapter"}),
    ("POST", "/v1/unload_lora_adapter", {"lora_name": "decision"}),
    ("POST", "/adapters", {"name": "copy", "src": "/pkg/adapter"}),
    ("DELETE", "/adapters/decision", None),
])
def test_runtime_lora_management_is_internal_only(method, path, body):
    app, owner, _, _ = make_app()
    with TestClient(app) as client:
        assert client.request(method, path, json=body).status_code == 403
        internal = {"x-systemone-internal": owner.settings.internal_token}
        assert client.request(method, path, json=body, headers=internal).status_code == 200


@pytest.mark.parametrize("header", ["X-Amzn-SageMaker-Adapter-Identifier", "X-Amzn-SageMaker-Adapter-Alias"])
def test_sagemaker_adapter_headers_cannot_select_the_adapter(header):
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        response = client.post("/invocations", json={"model": "base-a"}, headers={header: "decision"})
        assert response.status_code == 400
        assert client.post("/invocations", json={"model": "base-a"}).json()["received"]["model"] == "base-a"


def test_tls_request_without_self_url_explains_the_setting():
    app, owner, _, _ = make_app()
    owner.check_transport = CountingTransport(owner.check_transport)
    with TestClient(app, base_url="https://testserver") as client:
        for _ in range(2):  # a missing setting is answered at once, without retry hints or caching
            response = client.post("/v1/systemone", json=decision())
            assert response.status_code == 500 and "SYSTEMONE_SELF_URL" in response.text
            assert "retry-after" not in response.headers
    assert owner.check_transport.calls == 0


def test_failed_discovery_is_remembered_briefly(monkeypatch):
    app, owner, engine, _ = make_app(cards=[{"id": "base-a", "parent": None}])
    owner.check_transport = CountingTransport(owner.check_transport)
    with TestClient(app) as client:
        for _ in range(3):
            response = client.post("/v1/systemone", json=decision())
            assert response.status_code == 503 and response.headers["retry-after"] == "5"
        assert owner.check_transport.calls == 1
        monkeypatch.setattr(middleware, "BIND_RETRY_SECONDS", 0)
        assert client.post("/v1/systemone", json=decision()).status_code == 503
        assert owner.check_transport.calls == 2
    assert not engine.requests


def test_routed_packages_read_base_rows_under_the_served_name():
    engine = Engine()
    engine.base_row_types = {"noul"}
    engine.base_alias = "unbound"
    app, _, _, _ = make_app(engine=engine)
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=decision("base-b", "noul")).status_code == 200
    assert engine.base_alias == "base-a" and engine.backend_url.startswith("http://")


def test_build_engine_removes_its_temporary_metadata(tmp_path, monkeypatch):
    from tests.test_launch import package
    from bielik_one import launch
    from bielik_one.api import remote_vllm

    package("llama-adapter", tmp_path)
    monkeypatch.setenv("SYSTEMONE_HEADS_MANIFEST", str(tmp_path / "heads.json"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    written = []
    original = launch.frontend

    def frontend(*args, **kwargs):
        argv = original(*args, **kwargs)
        written.append(Path(argv[argv.index("--metadata") + 1]))
        return argv

    monkeypatch.setattr(launch, "frontend", frontend)
    with patch.object(remote_vllm, "RemoteVllmEngine",
                      side_effect=lambda metadata, url, **kw: SimpleNamespace(aliases=set(), client=kw["client"])), \
            patch.object(remote_vllm, "resolve_calibration", return_value=({}, None)):
        engine = middleware.build_engine(middleware.Settings(package=tmp_path, workers=3))
    engine.client.close()
    assert written and not written[0].parent.exists()


def test_decision_health_reports_readiness_and_needs_the_key():
    cards = [{"id": "base-a", "parent": None}]
    app, _, _, _ = make_app(keys=["key"], cards=cards)
    auth = {"Authorization": "Bearer key"}
    with TestClient(app) as client:
        assert client.get("/v1/systemone/health").status_code == 401
        response = client.get("/v1/systemone/health", headers=auth)
        assert response.status_code == 503 and "does not serve the decision adapter" in response.text
        assert "retry-after" not in response.headers
        cards.append({"id": "decision", "parent": "base-a"})
        middleware.BIND_RETRY_SECONDS, saved = 0, middleware.BIND_RETRY_SECONDS
        try:
            response = client.get("/v1/systemone/health", headers=auth)
        finally:
            middleware.BIND_RETRY_SECONDS = saved
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "models": ["base-a"], "calibrated": True,
                                   "calibration": "calibration.json"}
        cards.pop()  # the adapter was unloaded after binding
        assert client.get("/v1/systemone/health", headers=auth).status_code == 503


def test_existing_health_route_is_not_replaced():
    app = FastAPI()
    app.add_api_route("/v1/systemone/health", lambda: {"native": True}, methods=["GET"])
    with patch.object(middleware, "build_engine") as build:
        with pytest.raises(RuntimeError, match="already defines /v1/systemone/health"):
            middleware.SystemOneMiddleware(app, settings=middleware.Settings())
        build.assert_not_called()


def test_self_ca_verifies_loopback_reads(tmp_path):
    import certifi
    import ssl
    from bielik_one.api import reasoning

    with pytest.raises(RuntimeError, match="SYSTEMONE_SELF_CA"):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_SELF_CA": str(tmp_path / "no")})
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_SELF_CA": certifi.where()})
    context = settings.ssl_context()
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED
    assert middleware.Settings().ssl_context() is None
    app, owner, engine, _ = make_app(self_ca=certifi.where())
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=decision()).status_code == 200
    reasoner = engine.reasoners["original"]
    assert reasoner.tls_opener is not None
    assert reasoning.EndpointReasoner("http://x/v1", "m").tls_opener is None


def test_rebuilt_middleware_stack_reuses_the_first_instance():
    """vLLM 0.32's SageMaker bootstrap builds the middleware stack before Starlette builds it again."""
    app, owner, engine, _ = make_app(keys=["key"])
    app.middleware_stack = app.build_middleware_stack()  # a second SystemOneMiddleware over the same router
    second = app.middleware_stack
    while not isinstance(second, middleware.SystemOneMiddleware):
        second = second.app
    assert second is not owner and second.owner is owner
    assert [getattr(r, "path", None) for r in app.router.routes].count("/v1/systemone") == 1
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer key"}
        assert client.post("/v1/systemone", json=decision(), headers=headers).status_code == 200
        assert client.get("/v1/systemone/health", headers=headers).status_code == 200
        assert client.post("/v1/chat/completions", json={"model": "decision"}, headers=headers).status_code == 400
    assert engine.client.headers["Authorization"] == "Bearer key" and engine.closed == 1


def test_loopback_reads_verify_with_the_system_store_by_default():
    import ssl

    context = middleware.Settings().verify()
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_vllm_builtin_structured_decisions_route_is_replaced():
    """vLLM 0.32 always registers its own POST /v1/systemone, disabled (501) for SystemOne architectures."""
    app, _, engine, _ = make_app(native_route=True)
    assert [getattr(r, "path", None) for r in app.router.routes].count("/v1/systemone") == 1
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=decision())
        assert response.status_code == 200 and response.json()["answers"]["q"]["type"] == "choice"
    assert len(engine.requests) == 1


def test_token_in_requests_cannot_set_backend_flags():
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        body = {"model": "base-a", "token_ids": [1, 2],
                "sampling_params": {"max_tokens": 1, "extra_args": {"systemone_head": "base"}}}
        received = client.post("/inference/v1/generate", json=body).json()["received"]
        assert received["sampling_params"] == {"max_tokens": 1}
        for params in ({"prompt_logprobs": 1}, {"prompt_logprob_token_ids": [[1]]}, {"prompt_logprob_start": 0}):
            body["sampling_params"] = params
            assert client.post("/inference/v1/generate", json=body).status_code == 400
        body = {"model": "decision", "token_ids": [1]}
        assert client.post("/inference/v1/generate", json=body).status_code == 400
