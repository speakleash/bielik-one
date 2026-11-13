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
- /v1/chat/completions, /v1/completions and the other vLLM routes: the unchanged base model;
- GET /v1/models: the decision adapter's LoRA name is left out.

The adapter's LoRA name is internal. Client requests that name it are refused (400), `vllm_xargs` is removed from
client requests and prompt logprobs are refused, as in `bielik-one frontend`. SageMaker adapter headers and the
runtime LoRA management routes (VLLM_ALLOW_RUNTIME_LORA_UPDATING) are refused for clients (403), so the decision
adapter can be neither selected, unloaded nor loaded again under another name. Only this middleware's own loopback
reads, which carry a per-process secret header, use the adapter.

Environment:
- SYSTEMONE_PACKAGE (required): the adapter package directory; its checksums are verified at startup.
- SYSTEMONE_MODEL_NAME: optional; restricts /v1/systemone to this one of the base model's served names (default:
  every name vLLM serves the base model under, i.e. all --served-model-name values).
- SYSTEMONE_LORA_NAME: the adapter's --lora-modules name (default: decision, as printed by `bielik-one backend`).
- SYSTEMONE_SELF_URL: how this process reaches its own vLLM routes (default: the address the first decision request
  arrived on, with http://; required for --uds or TLS).
- SYSTEMONE_NO_CALIBRATION=1: raw temperature-1 probabilities instead of the package's calibration.json.
- SYSTEMONE_WORKERS: concurrent backend reads (default: the package's `bielik-one frontend` value, which matches its
  --max-num-seqs profile).
- SYSTEMONE_MAX_BODY_MB: largest decision or public request body (default 16); form uploads are not limited here.
- SYSTEMONE_INTERNAL_TOKEN: the secret of the loopback reads (default: random per process). With
  --api-server-count > 1 (vLLM's default with --data-parallel-size > 1) a loopback read can reach another API
  process, so a shared value is required then (the server refuses to start without it).
- Authentication and CORS: decisions pass through vLLM's own middleware, including API keys from its CLI or
  configuration file. Loopback reads use a configured server key (VLLM_API_KEY is the fallback).

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
import time

DECISION_PATH = "/v1/systemone"
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
    internal_token: str = field(default_factory=lambda: secrets.token_hex(16))
    shared_token: bool = False  # True when SYSTEMONE_INTERNAL_TOKEN sets one secret for every API process
    api_key: str | None = None
    calibration: bool = True
    workers: int | None = None  # None: the value `bielik-one frontend` uses for the package
    max_body_mb: float = 16

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
        settings = cls(
            public_name=public,
            lora_name=env.get("SYSTEMONE_LORA_NAME") or DEFAULT_LORA_NAME,
            package=root,
            self_url=(env.get("SYSTEMONE_SELF_URL") or "").rstrip("/") or None,
            api_key=env.get("VLLM_API_KEY") or None,
            calibration=not _flag(env.get("SYSTEMONE_NO_CALIBRATION")),
            workers=workers,
            max_body_mb=max_body_mb,
        )
        if env.get("SYSTEMONE_INTERNAL_TOKEN"):
            settings.internal_token = env["SYSTEMONE_INTERNAL_TOKEN"]
            settings.shared_token = True
        if settings.public_name is not None and settings.lora_name == settings.public_name:
            raise RuntimeError("SystemOne middleware: the LoRA name must differ from the public model name")
        if settings.workers is not None and not 1 <= settings.workers <= 256:
            raise RuntimeError("SYSTEMONE_WORKERS must be between 1 and 256")
        if not math.isfinite(settings.max_body_mb) or settings.max_body_mb * 2**20 < 1:
            raise RuntimeError("SYSTEMONE_MAX_BODY_MB must be finite and at least one byte")
        return settings


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


class _DecisionEndpoint:
    def __init__(self, owner):
        self.owner = owner

    async def __call__(self, scope, receive, send):
        return await self.owner._decide(scope, receive, send)


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
        if any(getattr(route, "path", None) == DECISION_PATH for route in router.routes):
            raise RuntimeError(f"vLLM already defines {DECISION_PATH}; refusing to shadow it")
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
        self._internal = self.settings.internal_token.encode()
        self._api_key = self.settings.api_key
        self._credentials_configured = False
        self.check_transport = None  # tests: an httpx transport for the startup self-check
        self._install_routes(router)

    def _install_routes(self, router):
        from starlette.routing import Route

        # A real route receives the same authentication, CORS, scaling and request-id
        # middleware as every other vLLM endpoint, including preflight requests.
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
            self._configure_credentials(scope)
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

    async def _decide(self, scope, receive, send):
        if not self._bound:
            failure = self._recent_failure()
            if failure is None:
                async with self._bind_lock:
                    # Requests that waited for a failed attempt answer from it instead of probing again.
                    failure = None if self._bound else (self._recent_failure() or await self._try_bind(scope))
            if failure is not None:
                _, status, body = failure
                retry = str(math.ceil(BIND_RETRY_SECONDS)).encode()
                return await _send_json(send, status, body, [(b"retry-after", retry)])
        return await self.decisions(scope, receive, send)

    async def _try_bind(self, scope):
        """Bind to this server's model names; on failure, the remembered (time, status, error body)."""
        from bielik_one.api.errors import BACKEND_UNAVAILABLE, INTERNAL_ERROR, error_bytes

        url = self.settings.self_url or self._url_from_scope(scope)
        if url is None:
            problem = (
                f"cannot derive this server's own URL from a {scope.get('scheme', 'unknown')!r} request "
                "(TLS or a Unix socket); set SYSTEMONE_SELF_URL"
            )
            status, code = 500, INTERNAL_ERROR
        else:
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
            async with httpx.AsyncClient(timeout=10, trust_env=False, transport=self.check_transport) as client:
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
            api_key=self._api_key,
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
            body = public_request_body(raw, {self.settings.lora_name}, None)
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
