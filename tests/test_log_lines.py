"""A failed request is logged as exactly one line, whatever its path contains."""

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from bielik_one.api.http_guard import RequestGuard, guarded, loggable


@pytest.fixture
def records():
    seen = []

    class Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    logger = logging.getLogger("bielik_one.api.http_guard")
    handler = Collect()
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.INFO)
    yield seen
    logger.removeHandler(handler)
    logger.setLevel(level)


def fail_with(path, method="GET"):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def run():
        messages = [{"type": "http.request", "body": b"", "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            pass

        await RequestGuard(app)({"type": "http", "method": method, "path": path, "headers": []}, receive, send)

    asyncio.run(run())


@pytest.mark.parametrize(
    "path, shown",
    [
        ("/x\nFORGED request ok answered 200", "/x\\x0aFORGED request ok answered 200"),
        ("/x\r\nFORGED", "/x\\x0d\\x0aFORGED"),
        ("/x FORGED", "/x\\u2028FORGED"),  # a line separator for many log viewers
        ("/x\x85FORGED", "/x\\x85FORGED"),
        ("/\x1b[2J\x1b[31mred", "/\\x1b[2J\\x1b[31mred"),  # terminal escape sequences
        ("/x\\x0aliteral", "/x\\\\x0aliteral"),  # a literal backslash cannot pass for an escape
    ],
)
def test_control_characters_in_the_path_stay_on_one_line(records, path, shown):
    fail_with(path)
    assert len(records) == 1
    line = records[0]
    assert "\n" not in line and "\r" not in line and " " not in line and "\x85" not in line and "\x1b" not in line
    assert shown in line and line.endswith("answered 500")


def test_ordinary_paths_and_methods_stay_readable(records):
    fail_with("/v1/chat/completions")
    fail_with("/v1/systemone/zażółć", method="POST")
    assert records[0].endswith("GET /v1/chat/completions answered 500")
    assert records[1].endswith("POST /v1/systemone/zażółć answered 500")


def test_the_method_is_escaped_too(records):
    fail_with("/x", method="GET\nFORGED")
    assert len(records) == 1 and "GET\\x0aFORGED /x" in records[0]


def test_percent_encoded_newline_over_http(records):
    async def app(scope, receive, send):  # every path fails; Starlette routes would not match a path with a newline
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    client = TestClient(guarded(app, 1), raise_server_exceptions=False)  # no lifespan: the bare app has none
    assert client.get("/x%0aFORGED%0d%0aagain").status_code == 500
    lines = [r for r in records if r.startswith("request ")]
    assert len(lines) == 1
    assert "/x\\x0aFORGED\\x0d\\x0aagain" in lines[0]


def test_loggable_handles_any_value():
    assert loggable(None) == "None"
    assert loggable("\U000e0001") == "\\U000e0001"
    assert loggable("plain /path?q=1") == "plain /path?q=1"
