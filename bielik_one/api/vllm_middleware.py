"""The typed decision API inside `vllm serve`: one process, one port, one public model name.

    VLLM_PLUGINS=systemone_heads SYSTEMONE_HEADS_MANIFEST=$PKG/heads.json VLLM_USE_V2_MODEL_RUNNER=0 \\
    SYSTEMONE_PACKAGE=$PKG \\
    vllm serve speakleash/Bielik-11B-v3.1-Instruct --revision ... [--served-model-name NAME] \\
        --enable-lora --lora-modules systemone-decision=$PKG/adapter ... \\
        --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware

Everything a client sees uses vLLM's served model name of the base model (--served-model-name, by default the model
argument); the middleware reads it from /v1/models on the first decision request, so nothing has to repeat it:
- POST /v1/systemone: typed decisions (choice, noul, score, labels) through the decision adapter, answered by the
  same code as `bielik-one frontend` (prompting, chunked logprob reads, calibration, errors);
- /v1/chat/completions, /v1/completions and the other vLLM routes: the unchanged base model;
- GET /v1/models: the decision adapter's LoRA name is left out.

The adapter's LoRA name is internal. Client requests that name it are refused (400), `vllm_xargs` is removed from
client requests and prompt logprobs are refused, as in `bielik-one frontend`. Only this middleware's own loopback
reads, which carry a per-process secret header, use the adapter.

Environment:
- SYSTEMONE_PACKAGE (required): the adapter package directory; its checksums are verified at startup.
- SYSTEMONE_MODEL_NAME: optional; restricts /v1/systemone to this one of the base model's served names (default:
  every name vLLM serves the base model under, i.e. all --served-model-name values).
- SYSTEMONE_LORA_NAME: the adapter's --lora-modules name (default: systemone-decision).
- SYSTEMONE_SELF_URL: how this process reaches its own vLLM routes (default: the address the first decision request
  arrived on, with http://; set it for --uds or TLS).
- SYSTEMONE_NO_CALIBRATION=1: raw temperature-1 probabilities instead of the package's calibration.json.
- SYSTEMONE_WORKERS: concurrent backend reads per decision request pool (default 8, the --max-num-seqs profile).
- SYSTEMONE_MAX_BODY_MB: largest /v1/systemone body (default 16).
- SYSTEMONE_INTERNAL_TOKEN: the secret of the loopback reads (default: random per process). With
  --api-server-count > 1 a loopback read can reach another API process, so set one shared value then.
- VLLM_API_KEY: vLLM's own key variable; when set, /v1/systemone requires it as well and loopback reads send it.
  `--api-key` on the command line is refused: the middleware cannot read it, so use VLLM_API_KEY.

Created by Filip Reka, filip.reka@cyfronet.pl
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import sys

DECISION_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
DEFAULT_LORA_NAME = "systemone-decision"
INTERNAL_HEADER = "x-systemone-internal"

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
    api_key: str | None = None
    calibration: bool = True
    workers: int = 8
    max_body_mb: float = 16

    @classmethod
    def from_env(cls, env=None, argv=None) -> "Settings":
        env = os.environ if env is None else env
        argv = sys.argv if argv is None else argv
        if any(arg == "--api-key" or arg.startswith("--api-key=") for arg in argv):
            raise RuntimeError(
                "SystemOne middleware: pass the API key as VLLM_API_KEY, not --api-key, "
                "so that /v1/systemone is protected by it too"
            )
        package = env.get("SYSTEMONE_PACKAGE")
        if not package:
            raise RuntimeError("SystemOne middleware: set SYSTEMONE_PACKAGE to the adapter package directory")
        root = Path(package).expanduser().resolve()
        public = env.get("SYSTEMONE_MODEL_NAME") or None
        settings = cls(
            public_name=public,
            lora_name=env.get("SYSTEMONE_LORA_NAME") or DEFAULT_LORA_NAME,
            package=root,
            self_url=(env.get("SYSTEMONE_SELF_URL") or "").rstrip("/") or None,
            api_key=env.get("VLLM_API_KEY") or None,
            calibration=not _flag(env.get("SYSTEMONE_NO_CALIBRATION")),
            workers=int(env.get("SYSTEMONE_WORKERS") or 8),
            max_body_mb=float(env.get("SYSTEMONE_MAX_BODY_MB") or 16),
        )
        if env.get("SYSTEMONE_INTERNAL_TOKEN"):
            settings.internal_token = env["SYSTEMONE_INTERNAL_TOKEN"]
        if settings.public_name is not None and settings.lora_name == settings.public_name:
            raise RuntimeError("SystemOne middleware: the LoRA name must differ from the public model name")
        if not 1 <= settings.workers <= 256:
            raise RuntimeError("SYSTEMONE_WORKERS must be between 1 and 256")
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
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("base_row_types"):
        # Routed packages read some question types from flagged base-model rows: under the served name, set at bind.
        metadata["base_alias"] = settings.public_name or "unbound"
    table, source = resolve_calibration(
        metadata_path, calibration_path, not settings.calibration, metadata.get("source_repo")
    )
    headers = {INTERNAL_HEADER: settings.internal_token}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    client = httpx.Client(
        timeout=180,
        trust_env=False,  # loopback: ignore HTTP_PROXY
        headers=headers,
        limits=httpx.Limits(max_connections=settings.workers, max_keepalive_connections=settings.workers),
    )
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
    # /v1/systemone answers to the served names only (not "decision" or the reference repo name); they are
    # read from vLLM when the middleware binds, so nothing is accepted before that.
    engine.aliases = set()
    log.info(
        "SystemOne decisions at %s (adapter %r, calibration: %s)", DECISION_PATH, settings.lora_name, source or "none"
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


class SystemOneMiddleware:
    """ASGI middleware for `vllm serve --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware`."""

    def __init__(self, app, settings: Settings | None = None, engine=None):
        from bielik_one.api.http_guard import guarded
        from bielik_one.api.remote_vllm import build_remote_app

        self.app = app
        self.settings = settings or Settings.from_env()
        self.engine = engine if engine is not None else build_engine(self.settings)
        # Only DECISION_PATH is routed here; the app's chat proxy and health routes are never reached.
        self.decisions = guarded(build_remote_app(self.engine), self.settings.max_body_mb)
        self.public_names: list[str] = []  # the base model's served names, read from vLLM at bind
        self._bound = False
        self._bind_lock = asyncio.Lock()
        self._internal = self.settings.internal_token.encode()
        self.check_transport = None  # tests: an httpx transport for the startup self-check

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = scope.get("headers", [])
        if self._is_internal(headers):
            return await self.app(scope, receive, send)
        path, method = scope.get("path", ""), scope.get("method", "")
        if path == DECISION_PATH:
            return await self._decide(scope, receive, send)
        if path == MODELS_PATH and method == "GET":
            return await self.app(scope, receive, self._models_filter(send))
        if method == "POST" and (_header(headers, b"content-type") or "").startswith("application/json"):
            return await self._guarded_post(scope, receive, send)
        return await self.app(scope, receive, send)

    def _is_internal(self, headers) -> bool:
        for key, value in headers:
            if key.lower() == INTERNAL_HEADER.encode():
                return hmac.compare_digest(value, self._internal)
        return False

    # -- /v1/systemone -------------------------------------------------------------------------------------------

    async def _decide(self, scope, receive, send):
        from bielik_one.api.errors import BACKEND_UNAVAILABLE, INTERNAL_ERROR, error_bytes

        if self.settings.api_key:
            given = _header(scope.get("headers", []), b"authorization") or ""
            if not hmac.compare_digest(given.encode(), f"Bearer {self.settings.api_key}".encode()):
                return await _send_json(send, 401, b'{"error":"Unauthorized"}')
        if not self._bound:
            async with self._bind_lock:
                if not self._bound:
                    url = self.settings.self_url or self._url_from_scope(scope)
                    if url is None:
                        body = error_bytes(500, INTERNAL_ERROR, "set SYSTEMONE_SELF_URL", path=DECISION_PATH)
                        return await _send_json(send, 500, body)
                    names, problem = await self._discover(url)
                    if problem:
                        log.error("SystemOne middleware: %s", problem)
                        body = error_bytes(503, BACKEND_UNAVAILABLE, problem, path=DECISION_PATH)
                        return await _send_json(send, 503, body)
                    self._bind(url, names)
        return await self.decisions(scope, receive, send)

    @staticmethod
    def _url_from_scope(scope) -> str | None:
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
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
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
                f"--lora-modules {lora}=<package>/adapter, or set SYSTEMONE_LORA_NAME to the name used there"
            )
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
            url + "/v1", names[0], key_env="VLLM_API_KEY", tok=getattr(self.engine, "tok", None)
        )
        self._bound = True
        log.info("SystemOne middleware: decisions as %s, read through %s", names, url)

    # -- the vLLM routes -----------------------------------------------------------------------------------------

    async def _guarded_post(self, scope, receive, send):
        """Client JSON requests: the LoRA name, vllm_xargs and prompt logprobs are refused or removed."""
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
        return await self.app(scope, replay, send)

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
