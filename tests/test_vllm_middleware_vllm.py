"""The middleware inside vLLM's own app (routes, auth and SageMaker bootstrap), without a GPU or an engine.

Skipped unless vLLM is installed (`pip install -e ".[vllm]"`). The unit tests in test_vllm_middleware.py use a small
FastAPI stand-in; this file catches what only vLLM's real app has (e.g. vLLM 0.32's own POST /v1/systemone).
"""

import pytest

pytest.importorskip("vllm")

from fastapi.testclient import TestClient  # noqa: E402

from bielik_one.api import vllm_middleware as middleware  # noqa: E402
from tests.test_vllm_middleware import Engine  # noqa: E402

MIDDLEWARE = "bielik_one.api.vllm_middleware.SystemOneMiddleware"


@pytest.fixture
def vllm_app(monkeypatch):
    from vllm.entrypoints.launchers import app as vllm_app_module
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    engines = []

    def build_engine(settings):
        engines.append(Engine())
        return engines[-1]

    for name in ("SYSTEMONE_LORA_NAME", "SYSTEMONE_MODEL_NAME", "SYSTEMONE_SELF_URL", "SYSTEMONE_SELF_CA",
                 "SYSTEMONE_WORKERS", "SYSTEMONE_MAX_BODY_MB", "SYSTEMONE_INTERNAL_TOKEN", "VLLM_API_KEY",
                 "VLLM_ALLOW_RUNTIME_LORA_UPDATING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SYSTEMONE_PACKAGE", "/nonexistent-package")
    monkeypatch.setattr(middleware, "build_engine", build_engine)
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        ["--model", "Qwen/Qwen3.5-4B", "--served-model-name", "qwen-base", "--api-key", "key",
         "--middleware", MIDDLEWARE]
    )
    app = vllm_app_module.build_app(args, ("generate",), None)  # its SageMaker bootstrap builds the stack
    app.middleware_stack = app.build_middleware_stack()  # and Starlette may build it again
    owner = app.middleware_stack
    while not isinstance(owner, middleware.SystemOneMiddleware):
        owner = owner.app
    return app, owner.owner, engines


def test_middleware_installs_in_vllms_app(vllm_app):
    app, owner, engines = vllm_app
    assert len(engines) == 1  # built once, although the middleware stack was built twice
    routes = [route for route in app.router.routes if getattr(route, "path", None) == middleware.DECISION_PATH]
    assert len(routes) == 1 and isinstance(routes[0].endpoint, middleware._DecisionEndpoint)
    paths = {getattr(route, "path", None): route for route in app.router.routes}
    assert middleware.HEALTH_PATH in paths
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/invocations"):
        assert isinstance(paths[path].app, middleware._PublicRoute), path
    assert isinstance(paths["/v1/models"].app, middleware._ModelListRoute)


def test_vllm_auth_and_public_filters_apply(vllm_app):
    app, owner, _ = vllm_app
    client = TestClient(app)  # no lifespan: the requests below never reach vLLM's engine
    key = {"Authorization": "Bearer key"}
    decision = {"model": "qwen-base", "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert client.post("/v1/systemone", json=decision).status_code == 401
    assert client.get("/v1/systemone/health").status_code == 401
    chat = {"model": "decision", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/v1/chat/completions", json=chat, headers=key)
    assert response.status_code == 400 and "answers typed decisions" in response.text
    # vLLM itself answers a text/plain body with 400 too: the middleware's own message must be the reason.
    response = client.post("/v1/completions", content=b'{"model": "decision", "prompt": "x"}',
                           headers={**key, "content-type": "text/plain"})
    assert response.status_code == 400 and "answers typed decisions" in response.text
    response = client.post("/invocations", json={**chat, "model": "qwen-base"},
                           headers={"X-Amzn-SageMaker-Adapter-Identifier": "decision"})
    assert response.status_code == 400 and "SageMaker" in response.text
    assert owner._api_key == "key" and owner.engine.client.headers["Authorization"] == "Bearer key"
