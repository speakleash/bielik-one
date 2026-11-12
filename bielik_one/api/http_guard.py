"""The outermost layer of a frontend: a request id for every request and a cap on the request body.

- Request id: a client's `X-Request-Id` (1-128 characters of letters, digits and `._:-`) is kept, anything else is
  replaced by a fresh `req-<hex>`. It is answered in the `x-request-id` header of every response (errors included),
  logged with failures, and available to the request's work through `REQUEST_ID`, so backend calls can carry it.
- Body cap: a body larger than `max_body_bytes` is refused with 413 before it is read whole: at once when its
  Content-Length says so, else as soon as the received bytes pass the cap. A body within the cap is buffered and
  replayed to the app unchanged.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets

from bielik_one.api.errors import FORMATS, PAYLOAD_TOO_LARGE, error_bytes

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
DEFAULT_MAX_BODY_MB = 16
_VALID_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

log = logging.getLogger(__name__)


class _Disconnected(Exception):
    """The client went away before its body was read whole."""


def request_id_from(headers) -> str:
    for key, value in headers:
        if key.lower() == b"x-request-id":
            text = value.decode("latin-1")
            if _VALID_ID.fullmatch(text):
                return text
            break
    return "req-" + secrets.token_hex(12)


def backend_request_id() -> str | None:
    """An id for one backend call of the current request: the request id plus a random part, so concurrent calls
    (and two clients that send the same id) never share an id in vLLM."""
    rid = REQUEST_ID.get()
    return None if rid is None else f"{rid}-{secrets.token_hex(8)}"


class RequestGuard:
    def __init__(self, app, max_body_bytes: int = DEFAULT_MAX_BODY_MB * 2**20, error_format: str = "default"):
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.error_format = error_format

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = request_id_from(scope.get("headers", []))
        token = REQUEST_ID.set(rid)
        status = None

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                message = {**message, "headers": headers + [(b"x-request-id", rid.encode())]}
            await send(message)

        try:
            try:
                body = await self._read_body(scope, receive)
            except _Disconnected:
                return  # nobody is left to answer
            if body is None:
                return await self._too_large(scope, send_with_id)
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, send_with_id)
        finally:
            # One line per failed request, with its id; the traceback of an unhandled error is uvicorn's.
            if status is None or status >= 500:
                log.warning(
                    "request %s %s %s answered %s", rid, scope.get("method"), scope.get("path"), status or "nothing"
                )
            REQUEST_ID.reset(token)

    async def _read_body(self, scope, receive) -> bytes | None:
        """The whole body, or None when it is larger than the cap (then it is not read any further)."""
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    if int(value) > self.max_body_bytes:
                        return None
                except ValueError:
                    pass
        parts, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _Disconnected
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_bytes:
                return None
            parts.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(parts)

    async def _too_large(self, scope, send):
        body = error_bytes(
            413,
            PAYLOAD_TOO_LARGE,
            f"request body is larger than {self.max_body_bytes} bytes",
            path=scope.get("path"),
            fmt=self.error_format,
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def max_body_mb(text: str) -> float:
    """argparse type: a finite size of at least one byte."""
    import argparse
    import math

    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or int(value * 2**20) < 1:
        raise argparse.ArgumentTypeError("must be a finite size of at least one byte")
    return value


def add_guard_arguments(parser):
    parser.add_argument(
        "--max-body-mb",
        type=max_body_mb,
        default=DEFAULT_MAX_BODY_MB,
        help=f"largest request body in MiB; larger ones get HTTP 413 (default: {DEFAULT_MAX_BODY_MB})",
    )
    parser.add_argument(
        "--error-format",
        choices=FORMATS,
        default="default",
        help='error bodies: "default" ({"error", "code"}) or "openai" ({"error": {"message", "type", "param", '
        '"code"}}); the chat routes always answer in the OpenAI format',
    )


def guarded(app, max_body_mb: float, error_format: str = "default") -> RequestGuard:
    return RequestGuard(app, int(max_body_mb * 2**20), error_format)
