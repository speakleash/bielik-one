"""One error body for every route of a frontend.

Default format (decision routes, `/v1/models`, unknown paths; `/health` keeps its `status` and adds these two):

    {"error": "<message>", "code": "<CODE>"}                      plus "detail" (field locations) on a 422

OpenAI format: the shape vLLM's OpenAI server answers (and OpenAI clients parse), always used on the chat routes,
which vLLM itself answers too, and on every route with `--error-format openai`:

    {"error": {"message": "<message>", "type": "invalid_request_error" | "server_error", "param": <field|null>,
               "code": <HTTP status>}}

Bodies are ASCII JSON, so text the client sent (even a lone surrogate) can always be written back.
"""

from __future__ import annotations

import json
from typing import Any

FORMATS = ("default", "openai")
OPENAI_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})

INVALID_REQUEST = "INVALID_REQUEST"  # 422: the request must be changed
NOT_FOUND = "NOT_FOUND"  # 404
METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"  # 405
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"  # 413
BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"  # 502 (503 on /health): vLLM did not answer usefully
TIMEOUT = "TIMEOUT"  # 504: vLLM did not answer in time
INTERNAL_ERROR = "INTERNAL_ERROR"  # 500
UNAUTHORIZED = "UNAUTHORIZED"  # 401
FORBIDDEN = "FORBIDDEN"  # 403
RATE_LIMITED = "RATE_LIMITED"  # 429

_BY_STATUS = {
    401: UNAUTHORIZED,
    403: FORBIDDEN,
    404: NOT_FOUND,
    405: METHOD_NOT_ALLOWED,
    413: PAYLOAD_TOO_LARGE,
    429: RATE_LIMITED,
    504: TIMEOUT,
}
_UNPARSABLE_BODY = "There was an error parsing the body"  # FastAPI's HTTPException(400) for a body it cannot read


def code_for_status(status: int) -> str:
    if status in _BY_STATUS:
        return _BY_STATUS[status]
    return INTERNAL_ERROR if status >= 500 else INVALID_REQUEST


def error_body(
    status: int,
    code: str,
    message: str,
    *,
    path: str | None = None,
    fmt: str = "default",
    detail: list[dict] | None = None,
) -> dict[str, Any]:
    if fmt == "openai" or path in OPENAI_PATHS:
        param = None
        if detail:
            loc = list(detail[0].get("loc", []))[1:]  # after "body" / "query" / ...
            if loc and all(isinstance(p, (str, int)) for p in loc) and isinstance(loc[0], str):
                param = ".".join(str(p) for p in loc)  # a field; a bare int is a parser's byte offset
        return {
            "error": {
                "message": message,
                "type": "server_error" if status >= 500 else "invalid_request_error",
                "param": param,
                "code": status,
            }
        }
    body: dict[str, Any] = {"error": message, "code": code}
    if detail:
        body["detail"] = detail
    return body


def error_bytes(status: int, code: str, message: str, **kwargs) -> bytes:
    return json.dumps(error_body(status, code, message, **kwargs), ensure_ascii=True).encode()


def error_response(status: int, code: str, message: str, *, headers: dict | None = None, **kwargs):
    from starlette.responses import Response

    return Response(
        error_bytes(status, code, message, **kwargs),
        status_code=status,
        media_type="application/json",
        headers=headers,
    )


def validation_detail(errors) -> list[dict]:
    """Field locations of a validation error, without the echoed input (a state can be 400,000 characters)."""
    return [
        {"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": str(e.get("type", "value_error"))}
        for e in errors
    ]


def install(app, fmt: str = "default") -> None:
    """Route every error of a FastAPI app through error_body."""
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException

    if fmt not in FORMATS:
        raise ValueError(f"error format must be one of {FORMATS}")
    app.state.error_format = fmt

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        detail = validation_detail(exc.errors())
        message = detail[0]["msg"] if detail else "invalid request"
        return error_response(
            422, INVALID_REQUEST, message, path=request.url.path, fmt=fmt, detail=detail
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        status = exc.status_code
        if status == 400 and exc.detail == _UNPARSABLE_BODY:
            status = 422  # a body FastAPI could not parse is a request to change, like any other
        detail = exc.detail if isinstance(exc.detail, list) else None
        if detail:
            message = detail[0].get("msg", "invalid request")
        elif isinstance(exc.detail, str):
            message = exc.detail
        else:
            message = json.dumps(exc.detail, ensure_ascii=True, default=str)
        return error_response(
            status,
            code_for_status(status),
            message,
            path=request.url.path,
            fmt=fmt,
            detail=detail,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        return error_response(
            500, INTERNAL_ERROR, "internal server error", path=request.url.path, fmt=fmt
        )
