"""The typed decision API inside `vllm serve`: one API server, one port, one public model name.

    VLLM_PLUGINS=systemone_heads SYSTEMONE_HEADS_MANIFEST=$PKG/heads.json VLLM_USE_V2_MODEL_RUNNER=0 \\
    SYSTEMONE_PACKAGE=$PKG \\
    vllm serve speakleash/Bielik-11B-v3.1-Instruct --revision ... [--served-model-name NAME] \\
        --enable-lora --lora-modules decision=$PKG/adapter ... \\
        --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware

`bielik-one backend $PKG --shell` prints the full validated command (the LoRA path is $PKG/adapter for Llama adapter
packages and $PKG for Qwen ones); append the --middleware flag to it.

Everything a client sees uses vLLM's served model name of the base model (--served-model-name, by default the model
argument); the middleware reads it from /v1/models on the first decision request, so nothing has to repeat it:
- POST /v1/systemone: typed decisions (choice, noul, score, labels) through the decision adapter, answered by the
  same code as `bielik-one frontend` (prompting, chunked logprob reads, calibration, errors);
- GET /v1/systemone/health: readiness of decisions (200, or 503/500 with the reason): the server's own model list is
  read and must still contain the decision adapter. vLLM's /health only reports the engine;
- /v1/chat/completions, /v1/completions and the other vLLM routes: the unchanged base model;
- GET /v1/models: the decision adapter's LoRA name is left out.

vLLM >= 0.32 registers its own POST /v1/systemone (generic structured decisions for a few stock architectures,
disabled with 501 for the SystemOne architectures of adapter packages); the middleware replaces that route only.

The adapter's LoRA name is internal. Client requests that name it are refused (400), `vllm_xargs` is removed from
client requests (and `extra_args` from token-in `sampling_params`) and prompt logprobs are refused, as in
`bielik-one frontend`. SageMaker adapter headers and the
runtime LoRA management routes (VLLM_ALLOW_RUNTIME_LORA_UPDATING) are refused for clients (403), so the decision
adapter can be neither selected, unloaded nor loaded again under another name. Only this middleware's own loopback
reads, which carry a per-process secret header, use the adapter.

Environment:
- SYSTEMONE_PACKAGE (required): the adapter package directory; its checksums are verified at startup.
- SYSTEMONE_MODEL_NAME: optional; restricts /v1/systemone to this one of the base model's served names (default:
  every name vLLM serves the base model under, i.e. all --served-model-name values).
- SYSTEMONE_LORA_NAME: the adapter's --lora-modules name (default: decision, as printed by `bielik-one backend`).
- SYSTEMONE_SELF_URL: how this process reaches its own vLLM routes (default: the address the first decision request
  arrived on, with http://; required for --uds or TLS, e.g. https://<name in the certificate>:8000).
- SYSTEMONE_SELF_CA: a CA bundle (PEM) that verifies the TLS certificate of SYSTEMONE_SELF_URL, when it is not
  signed by a CA in the system trust store (proxy variables are ignored for loopback reads).
- SYSTEMONE_NO_CALIBRATION=1: raw temperature-1 probabilities instead of the package's calibration.json.
- SYSTEMONE_WORKERS: concurrent backend reads (default: the package's `bielik-one frontend` value, which matches its
  --max-num-seqs profile).
- SYSTEMONE_MAX_BODY_MB: largest decision or public request body (default 16); form uploads are not limited here.
- SYSTEMONE_MAX_REQUEST_TOKENS: prompt tokens one decision request may send in all, counted on the standard prompts
  (default 4194304); larger requests get 422, as with `bielik-one frontend --max-request-tokens`.
- SYSTEMONE_REQUEST_TIMEOUT: seconds one decision request may spend on its reads (default 600); slower ones get 504.
- SYSTEMONE_INTERNAL_TOKEN: the secret of the loopback reads (default: random per process). With
  --api-server-count > 1 (vLLM's default with --data-parallel-size > 1) a loopback read can reach another API
  process, so a shared value is required then (the server refuses to start without it).
- Authentication and CORS: decisions pass through vLLM's own middleware, including API keys from its CLI or
  configuration file. Loopback reads use a configured server key (VLLM_API_KEY is the fallback). vLLM's API key
  guards only paths under /v1: /invocations, /health, /metrics and the other routes stay open, as in plain vLLM.

Created by Filip Reka, filip.reka@cyfronet.pl
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import json
import logging
import math
import os
from pathlib import Path
import secrets
import shutil
import ssl
import time

DECISION_PATH = "/v1/systemone"
HEALTH_PATH = "/v1/systemone/health"
MODELS_PATH = "/v1/models"
DEFAULT_LORA_NAME = "decision"
INTERNAL_HEADER = "x-systemone-internal"
# Runtime LoRA management (VLLM_ALLOW_RUNTIME_LORA_UPDATING), OpenAI-style and SageMaker-style: a client could unload
# the decision adapter or load it again under a public name, so these routes are internal-only.
LORA_ADMIN_PATHS = ("/v1/load_lora_adapter", "/v1/unload_lora_adapter", "/adapters")
# SageMaker routes (/invocations) copy these headers into the request's model after our body filter has run.
ADAPTER_HEADERS = (b"x-amzn-sagemaker-adapter-identifier", b"x-amzn-sagemaker-adapter-alias")
FORM_MEDIA = ("multipart/form-data", "application/x-www-form-urlencoded")
BIND_RETRY_SECONDS = 5.0  # a failed setup check is answered from memory this long, instead of re-probing per request

log = logging.getLogger(__name__)


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    public_name: str | None = None  # None: the base model's served names, read from vLLM
    lora_name: str = DEFAULT_LORA_NAME
    package: Path | None = None
    self_url: str | None = None
    self_ca: str | None = None  # PEM bundle verifying SYSTEMONE_SELF_URL's certificate
    internal_token: str = field(default_factory=lambda: secrets.token_hex(16))
    shared_token: bool = False  # True when SYSTEMONE_INTERNAL_TOKEN sets one secret for every API process
    api_key: str | None = None
    calibration: bool = True
    workers: int | None = None  # None: the value `bielik-one frontend` uses for the package
    max_body_mb: float = 16
    max_request_tokens: int = 4_194_304  # remote_vllm.DEFAULT_MAX_REQUEST_TOKENS
    request_timeout: float = 600.0  # remote_vllm.DEFAULT_REQUEST_TIMEOUT

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        package = env.get("SYSTEMONE_PACKAGE")
        if not package:
            raise RuntimeError("SystemOne middleware: set SYSTEMONE_PACKAGE to the adapter package directory")
        root = Path(package).expanduser().resolve()
        public = env.get("SYSTEMONE_MODEL_NAME") or None
        try:
            workers = int(env["SYSTEMONE_WORKERS"]) if env.get("SYSTEMONE_WORKERS") else None
            max_body_mb = float(env.get("SYSTEMONE_MAX_BODY_MB") or 16)
        except ValueError as exc:
            raise RuntimeError(f"SystemOne middleware: invalid SYSTEMONE_WORKERS or SYSTEMONE_MAX_BODY_MB ({exc})")
        try:
            max_request_tokens = int(env.get("SYSTEMONE_MAX_REQUEST_TOKENS") or cls.max_request_tokens)
            request_timeout = float(env.get("SYSTEMONE_REQUEST_TIMEOUT") or cls.request_timeout)
        except ValueError as exc:
            raise RuntimeError(
                f"SystemOne middleware: invalid SYSTEMONE_MAX_REQUEST_TOKENS or SYSTEMONE_REQUEST_TIMEOUT ({exc})"
            )
        settings = cls(
            public_name=public,
            lora_name=env.get("SYSTEMONE_LORA_NAME") or DEFAULT_LORA_NAME,
            package=root,
            self_url=(env.get("SYSTEMONE_SELF_URL") or "").rstrip("/") or None,
            self_ca=env.get("SYSTEMONE_SELF_CA") or None,
            api_key=env.get("VLLM_API_KEY") or None,
            calibration=not _flag(env.get("SYSTEMONE_NO_CALIBRATION")),
            workers=workers,
            max_body_mb=max_body_mb,
            max_request_tokens=max_request_tokens,
            request_timeout=request_timeout,
        )
        if env.get("SYSTEMONE_INTERNAL_TOKEN"):
            settings.internal_token = env["SYSTEMONE_INTERNAL_TOKEN"]
            settings.shared_token = True
        if settings.public_name is not None and settings.lora_name == settings.public_name:
            raise RuntimeError("SystemOne middleware: the LoRA name must differ from the public model name")
        if settings.self_ca is not None and not Path(settings.self_ca).expanduser().is_file():
            raise RuntimeError(f"SYSTEMONE_SELF_CA={settings.self_ca!r} is not a file")
        if settings.workers is not None and not 1 <= settings.workers <= 256:
            raise RuntimeError("SYSTEMONE_WORKERS must be between 1 and 256")
        if not math.isfinite(settings.max_body_mb) or settings.max_body_mb * 2**20 < 1:
            raise RuntimeError("SYSTEMONE_MAX_BODY_MB must be finite and at least one byte")
        if settings.max_request_tokens < 1:
            raise RuntimeError("SYSTEMONE_MAX_REQUEST_TOKENS must be at least 1")
        if not (math.isfinite(settings.request_timeout) and settings.request_timeout > 0):
            raise RuntimeError("SYSTEMONE_REQUEST_TIMEOUT must be a finite number of seconds greater than 0")
        return settings

    def ssl_context(self) -> ssl.SSLContext | None:
        """SYSTEMONE_SELF_CA's verification context; None means the system trust store (see `verify`)."""
        if self.self_ca is None:
            return None
        return ssl.create_default_context(cafile=str(Path(self.self_ca).expanduser()))

    def verify(self) -> ssl.SSLContext:
        """httpx verification of loopback reads: SYSTEMONE_SELF_CA, else the system trust store (as urllib uses),
        not httpx's bundled certifi list, so a CA installed on the host verifies as it does for the reasoning pass."""
        return self.ssl_context() or ssl.create_default_context()


def build_engine(settings: Settings):
    """The decision engine of `bielik-one frontend`, reading through this server's own routes under the LoRA name."""
    import httpx

    from bielik_one import launch
    from bielik_one.api.remote_vllm import RemoteVllmEngine, resolve_calibration

    root = settings.package
    launch.verify(root)
    kind = launch.detect(root)
    if kind not in ("llama-adapter", "qwen-adapter"):
        raise RuntimeError(f"{root}: a {kind} package; the middleware serves adapter packages (*-Adapter)")
    manifest = os.environ.get("SYSTEMONE_HEADS_MANIFEST")
    if not manifest or Path(manifest).resolve() != (root / "heads.json").resolve():
        raise RuntimeError(
            f"SYSTEMONE_HEADS_MANIFEST must be {root / 'heads.json'} (the heads of the package in SYSTEMONE_PACKAGE)"
        )
    # Same metadata as `bielik-one frontend` (tokenizer snapshot, pinned chat template, adapter, heads).
    argv = launch.frontend(root, kind, "127.0.0.1", 0, "http://unbound", not settings.calibration, dry_run=False)
    metadata_path = Path(argv[argv.index("--metadata") + 1])
    calibration_path = Path(argv[argv.index("--calibration") + 1]) if "--calibration" in argv else None
    try:
        metadata = json.loads(metadata_path.read_text())
        table, source = resolve_calibration(
            metadata_path, calibration_path, not settings.calibration, metadata.get("source_repo")
        )
    finally:
        # launch.frontend writes adapter metadata to a fresh temporary directory; everything needed is read now.
        tmp = metadata_path.parent
        if tmp.name.startswith("bielik-one-") and tmp != root and root not in tmp.parents:
            shutil.rmtree(tmp, ignore_errors=True)
    if settings.workers is None:
        settings.workers = int(argv[argv.index("--workers") + 1])
    if metadata.get("base_row_types"):
        # Routed packages read some question types from flagged base-model rows: under the served name, set at bind.
        metadata["base_alias"] = settings.public_name or "unbound"
    headers = {INTERNAL_HEADER: settings.internal_token}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    client = httpx.Client(
        timeout=180,
        trust_env=False,  # loopback: ignore HTTP_PROXY
        headers=headers,
        verify=settings.verify(),
        limits=httpx.Limits(max_connections=settings.workers, max_keepalive_connections=settings.workers),
    )
    try:
        engine = RemoteVllmEngine(
            metadata,
            "http://unbound",  # set from the first decision request (or SYSTEMONE_SELF_URL)
            model_name=settings.public_name or "unbound",
            backend_model=settings.lora_name,
            workers=settings.workers,
            calibration=table,
            calibration_source=source,
            client=client,
            max_request_tokens=settings.max_request_tokens,
            request_timeout=settings.request_timeout,
        )
    except BaseException:
        client.close()
        raise
    # /v1/systemone answers to the served names only (not "decision" or the reference repo name); they are
    # read from vLLM when the middleware binds, so nothing is accepted before that.
    engine.aliases = set()
    log.info(
        "SystemOne decisions at %s (adapter %r, %d workers, calibration: %s)",
        DECISION_PATH, settings.lora_name, settings.workers, source or "none",
    )
    return engine


def _header(headers, name: bytes) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _with_header(headers, name: bytes, value: bytes) -> list:
    return [(k, v) for k, v in headers if k.lower() != name] + [(name, value)]


async def _send_json(send, status: int, body: bytes, extra_headers=()):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *extra_headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive) -> bytes | None:
    """The whole request body, or None when the client went away."""
    parts = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        parts.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(parts)


def _router(app):
    """Find vLLM's router below its authentication/CORS middleware stack."""
    from fastapi import FastAPI
    from starlette.routing import Router

    seen = set()
    while app is not None and id(app) not in seen:
        seen.add(id(app))
        if isinstance(app, FastAPI):
            return app.router
        if isinstance(app, Router):
            return app
        app = getattr(app, "app", None)
    raise RuntimeError("SystemOne middleware requires vLLM's FastAPI router")


def _vllm_structured_decisions(route) -> bool:
    """vLLM's built-in /v1/systemone (>= 0.32: generic structured decisions for a few stock architectures). It is
    disabled (501) for the SystemOne architectures of adapter packages, so the middleware's route replaces it."""
    module = getattr(getattr(route, "endpoint", None), "__module__", None) or ""
    return module.startswith("vllm.entrypoints.generate.structured_decisions")


def _scrub_sampling_params(raw: bytes) -> bytes:
    """Token-in requests (/inference/v1/generate) carry full SamplingParams: `extra_args` would reach the decision-head
    logits processor like vllm_xargs, and prompt logprobs are not supported by it."""
    from bielik_one.api.server import RequestError

    try:
        body = json.loads(raw)
    except ValueError:
        return raw
    params = body.get("sampling_params") if isinstance(body, dict) else None
    if not isinstance(params, dict):
        return raw
    for key in ("prompt_logprobs", "prompt_logprob_token_ids", "prompt_logprob_start"):
        if params.get(key) is not None:
            raise RequestError(["sampling_params", key], "prompt logprobs are not supported by this server")
    if "extra_args" not in params:
        return raw
    params.pop("extra_args")
    return json.dumps(body).encode()


class _DecisionEndpoint:
    def __init__(self, owner):
        self.owner = owner

    async def __call__(self, scope, receive, send):
        return await self.owner._decide(scope, receive, send)


class _HealthEndpoint:
    def __init__(self, owner):
        self.owner = owner

    async def __call__(self, scope, receive, send):
        return await self.owner._health(scope, receive, send)


class _PublicRoute:
    """Filter client POST requests after vLLM's auth/CORS, whatever their declared content type.

    Bodies that are not JSON objects pass through unchanged for vLLM to answer; form uploads (e.g. audio
    transcriptions) are neither buffered nor size-limited here.
    """

    def __init__(self, app, owner):
        from bielik_one.api.http_guard import guarded

        self.app, self.owner = app, owner
        self.guarded = guarded(self._filtered, owner.settings.max_body_mb, "openai")

    async def _filtered(self, scope, receive, send):
        return await self.owner._guarded_post(scope, receive, send, next_app=self.app)

    async def __call__(self, scope, receive, send):
        from bielik_one.api.errors import INVALID_REQUEST, error_bytes

        headers = scope.get("headers", [])
        if scope.get("method") == "POST" and not self.owner._is_internal(headers):
            if any(key.lower() in ADAPTER_HEADERS for key, _ in headers):
                message = "SageMaker adapter headers are not accepted; use /v1/systemone for decisions"
                return await _send_json(send, 400, error_bytes(400, INVALID_REQUEST, message, fmt="openai"))
            media = (_header(headers, b"content-type") or "").split(";", 1)[0].strip().lower()
            if media not in FORM_MEDIA:
                return await self.guarded(scope, receive, send)
        return await self.app(scope, receive, send)


class _InternalOnlyRoute:
    """Routes clients must not use (runtime LoRA management): 403 unless the request is a loopback read."""

    def __init__(self, app, owner):
        self.app, self.owner = app, owner

    async def __call__(self, scope, receive, send):
        from bielik_one.api.errors import INVALID_REQUEST, error_bytes

        if self.owner._is_internal(scope.get("headers", [])):
            return await self.app(scope, receive, send)
        message = "runtime LoRA management is disabled while the SystemOne decision adapter is served"
        return await _send_json(send, 403, error_bytes(403, INVALID_REQUEST, message, fmt="openai"))


class _ModelListRoute:
    def __init__(self, app, owner):
        self.app, self.owner = app, owner

    async def __call__(self, scope, receive, send):
        if scope.get("method") == "GET" and not self.owner._is_internal(scope.get("headers", [])):
            send = self.owner._models_filter(send)
        return await self.app(scope, receive, send)


class SystemOneMiddleware:
    """ASGI middleware for `vllm serve --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware`."""

    def __init__(self, app, settings: Settings | None = None, engine=None):
        from bielik_one.api.http_guard import guarded
        from bielik_one.api.remote_vllm import install_backend_errors
        from bielik_one.api.server import build_app

        self.app = app
        router = _router(app)
        existing = getattr(router, "_systemone_owner", None)
        if existing is not None:
            # Starlette may build the middleware stack more than once (vLLM's SageMaker bootstrap does so before
            # serving): the routes, engine and state of the first instance stay in charge; this instance's
            # settings/engine arguments are ignored, and nothing is built for it.
            self.owner = existing
            return
        self.owner = self
        for path in (DECISION_PATH, HEALTH_PATH):
            if any(getattr(r, "path", None) == path and not _vllm_structured_decisions(r) for r in router.routes):
                raise RuntimeError(f"vLLM already defines {path}; refusing to shadow it")
        self.settings = settings or Settings.from_env()
        self.engine = engine if engine is not None else build_engine(self.settings)
        # Only the typed decision route: the chat, model and health routes are vLLM's own.
        self.decision_app = build_app(self.engine)
        install_backend_errors(self.decision_app)
        self.decisions = guarded(self.decision_app, self.settings.max_body_mb)
        self.public_names: list[str] = []  # the base model's served names, read from vLLM at bind
        self._bound = False
        self._bind_lock = asyncio.Lock()
        self._bind_failure: tuple[float, int, bytes] | None = None  # (monotonic time, status, error body)
        self._ssl = self.settings.ssl_context()
        self._verify = self.settings.verify()
        self._internal = self.settings.internal_token.encode()
        self._api_key = self.settings.api_key
        self._credentials_configured = False
        self.check_transport = None  # tests: an httpx transport for the startup self-check
        self._install_routes(router)
        router._systemone_owner = self

    def _install_routes(self, router):
        from starlette.routing import Route

        # A real route receives the same authentication, CORS, scaling and request-id
        # middleware as every other vLLM endpoint, including preflight requests.
        native = [r for r in router.routes if getattr(r, "path", None) == DECISION_PATH]
        if native:
            router.routes[:] = [r for r in router.routes if r not in native]
            log.info("SystemOne middleware: replacing vLLM's built-in %s (structured decisions)", DECISION_PATH)
        for route in router.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None) or set()
            if path and any(path == p or path.startswith(p + "/") for p in LORA_ADMIN_PATHS):
                route.app = _InternalOnlyRoute(route.app, self)
            elif path == MODELS_PATH and "GET" in methods:
                route.app = _ModelListRoute(route.app, self)
            elif "POST" in methods:
                route.app = _PublicRoute(route.app, self)
        router.routes.append(Route(DECISION_PATH, _DecisionEndpoint(self), methods=["POST"]))
        router.routes.append(Route(HEALTH_PATH, _HealthEndpoint(self), methods=["GET"]))
        original_lifespan = router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            # The engine's HTTP client and thread pool are released even when vLLM startup fails.
            try:
                self._check_server_count(app)
                async with original_lifespan(app) as state:
                    yield state
            finally:
                self.engine.close()

        router.lifespan_context = lifespan

    def _check_server_count(self, app):
        args = getattr(getattr(app, "state", None), "args", None)
        count = getattr(args, "api_server_count", None) or 1
        if count > 1 and not self.settings.shared_token:
            raise RuntimeError(
                f"SystemOne middleware: --api-server-count {count} needs SYSTEMONE_INTERNAL_TOKEN set to one secret "
                "shared by every API process (loopback reads can reach any of them)"
            )

    def _configure_credentials(self, scope):
        if self._credentials_configured:
            return
        state = getattr(scope.get("app"), "state", None)
        args = getattr(state, "args", None)
        keys = getattr(args, "api_key", None) or [self.settings.api_key]
        if isinstance(keys, str):
            keys = [keys]
        self._api_key = next((key for key in keys if key), None)
        if self._api_key:
            self.engine.client.headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            self.engine.client.headers.pop("Authorization", None)
        self._credentials_configured = True

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.owner._configure_credentials(scope)
        return await self.app(scope, receive, send)

    def _is_internal(self, headers) -> bool:
        for key, value in headers:
            if key.lower() == INTERNAL_HEADER.encode():
                return hmac.compare_digest(value, self._internal)
        return False

    # -- /v1/systemone -------------------------------------------------------------------------------------------

    def _recent_failure(self):
        if self._bind_failure and time.monotonic() - self._bind_failure[0] < BIND_RETRY_SECONDS:
            return self._bind_failure
        return None

    async def _ensure_bound(self, scope):
        """None once bound, else the failure (time, status, error body) to answer with."""
        if self._bound:
            return None
        failure = self._recent_failure()
        if failure is None:
            async with self._bind_lock:
                # Requests that waited for a failed attempt answer from it instead of probing again.
                failure = None if self._bound else (self._recent_failure() or await self._try_bind(scope))
        return failure

    @staticmethod
    async def _send_failure(send, status: int, body: bytes):
        # Only an unreachable or not yet ready server is worth retrying; a missing setting needs a restart.
        retry = [(b"retry-after", str(math.ceil(BIND_RETRY_SECONDS)).encode())] if status == 503 else []
        return await _send_json(send, status, body, retry)

    async def _decide(self, scope, receive, send):
        failure = await self._ensure_bound(scope)
        if failure is not None:
            return await self._send_failure(send, failure[1], failure[2])
        return await self.decisions(scope, receive, send)

    async def _health(self, scope, receive, send):
        """Readiness of /v1/systemone: bound, and the decision adapter is still in vLLM's model list."""
        from bielik_one.api.errors import BACKEND_UNAVAILABLE, error_bytes

        failure = await self._ensure_bound(scope)
        if failure is None:
            _, problem = await self._discover(self.engine.backend_url)
            if problem:
                log.error("SystemOne middleware: %s", problem)
                failure = (0.0, 503, error_bytes(503, BACKEND_UNAVAILABLE, problem, path=HEALTH_PATH))
        if failure is not None:
            return await _send_json(send, failure[1], failure[2])
        source = getattr(self.engine, "calibration_source", None)
        body = {
            "status": "ok",
            "models": self.public_names,
            "calibrated": bool(getattr(self.engine, "temperatures", None)),
            "calibration": Path(source).name if source else None,
        }
        return await _send_json(send, 200, json.dumps(body).encode())

    async def _try_bind(self, scope):
        """Bind to this server's model names; on failure, the remembered (time, status, error body)."""
        from bielik_one.api.errors import BACKEND_UNAVAILABLE, INTERNAL_ERROR, error_bytes

        url = self.settings.self_url or self._url_from_scope(scope)
        if url is None:
            problem = (
                f"cannot derive this server's own URL from a {scope.get('scheme', 'unknown')!r} request "
                "(TLS or a Unix socket); set SYSTEMONE_SELF_URL"
            )
            log.error("SystemOne middleware: %s", problem)
            # A configuration error: answered at once on every request, not remembered as a retryable failure.
            return (time.monotonic(), 500, error_bytes(500, INTERNAL_ERROR, problem, path=DECISION_PATH))
        names, problem = await self._discover(url)
        if not problem:
            self._bind(url, names)
            self._bind_failure = None
            return None
        status, code = 503, BACKEND_UNAVAILABLE
        log.error("SystemOne middleware: %s", problem)
        self._bind_failure = (time.monotonic(), status, error_bytes(status, code, problem, path=DECISION_PATH))
        return self._bind_failure

    @staticmethod
    def _url_from_scope(scope) -> str | None:
        """Plain HTTP over TCP only: a TLS server's certificate does not name its own socket address."""
        if scope.get("scheme", "http") != "http":
            return None
        server = scope.get("server")
        if not server or server[1] is None:
            return None
        host, port = server
        host = f"[{host}]" if ":" in host else host
        return f"http://{host}:{port}"

    async def _discover(self, url: str) -> tuple[list[str], str | None]:
        """The base model's served names from vLLM's own /v1/models, and a problem when the setup cannot work.

        vLLM lists the base model under each --served-model-name (in that order, without a "parent") and every LoRA
        adapter with its "parent". The decision adapter must be one of the adapters, or every read would fail later.
        """
        import httpx

        headers = {INTERNAL_HEADER: self.settings.internal_token}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(
                timeout=10, trust_env=False, verify=self._verify, transport=self.check_transport
            ) as client:
                response = await client.get(url + MODELS_PATH, headers=headers)
                response.raise_for_status()
                cards = [card for card in response.json()["data"] if isinstance(card, dict)]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return [], f"cannot list this server's models at {url}{MODELS_PATH} ({exc!r}); set SYSTEMONE_SELF_URL"
        lora = self.settings.lora_name
        served = [card.get("id") for card in cards]
        if lora not in served:
            return [], (
                f"vLLM does not serve the decision adapter {lora!r} (served: {served}); start it with "
                f"--lora-modules {lora}=<adapter path>, or set SYSTEMONE_LORA_NAME to the name used there"
            )
        if not any(card.get("id") == lora and card.get("parent") for card in cards):
            return [], f"{lora!r} is not a LoRA adapter in vLLM's model list"
        names = [card["id"] for card in cards if not card.get("parent") and card.get("id") and card["id"] != lora]
        if not names:
            return [], f"vLLM lists no base model name (served: {served})"
        wanted = self.settings.public_name
        if wanted is not None:
            if wanted not in names:
                return [], (
                    f"SYSTEMONE_MODEL_NAME={wanted!r} is not a served name of the base model (served: {names}); "
                    "unset it to use vLLM's --served-model-name"
                )
            names = [wanted]
        return names, None

    def _bind(self, url: str, names: list[str]):
        from bielik_one.api.reasoning import EndpointReasoner

        self.public_names = list(names)
        self.engine.aliases = set(names)
        self.engine.model_name = names[0]  # named in "use model ..." errors
        if getattr(self.engine, "base_row_types", None):
            self.engine.base_alias = names[0]
        self.engine.backend_url = url
        # Optional reasoning pass: generated by the base model under its served name, through the vLLM routes.
        self.engine.reasoners["original"] = EndpointReasoner(
            url + "/v1", names[0], key_env="VLLM_API_KEY", tok=getattr(self.engine, "tok", None),
            api_key=self._api_key, ssl_context=self._ssl,
        )
        self._bound = True
        log.info("SystemOne middleware: decisions as %s, read through %s", names, url)

    # -- the vLLM routes -----------------------------------------------------------------------------------------

    async def _guarded_post(self, scope, receive, send, *, next_app):
        """Client requests: the LoRA name, vllm_xargs and prompt logprobs are refused or removed."""
        from bielik_one.api.errors import INVALID_REQUEST, error_bytes
        from bielik_one.api.remote_vllm import public_request_body
        from bielik_one.api.server import RequestError

        raw = await _read_body(receive)
        if raw is None:
            return
        try:
            # A request without a model keeps none: vLLM answers it with its default served name, as without us.
            body = public_request_body(_scrub_sampling_params(raw), {self.settings.lora_name}, None)
        except RequestError as exc:
            return await _send_json(send, 400, error_bytes(400, INVALID_REQUEST, exc.msg, fmt="openai"))
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        if body is not raw:
            scope = dict(scope, headers=_with_header(scope["headers"], b"content-length", str(len(body)).encode()))
        return await next_app(scope, replay, send)

    def _models_filter(self, send):
        """Leave the LoRA name out of GET /v1/models."""
        start = None
        parts = []

        async def filtered(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] != "http.response.body" or start is None:
                return await send(message)
            parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            body = b"".join(parts)
            if start["status"] == 200:
                try:
                    listing = json.loads(body)
                    listing["data"] = [m for m in listing["data"] if m.get("id") != self.settings.lora_name]
                    body = json.dumps(listing).encode()
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
            headers = _with_header(start.get("headers", []), b"content-length", str(len(body)).encode())
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body})

        return filtered
