"""Typed decisions + masked chat over an ordinary plugin-free merged vLLM."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import httpx, uvicorn
from bielik_one.api.remote_vllm import (
    RemoteVllmEngine,
    add_calibration_arguments,
    add_limit_arguments,
    build_remote_app,
    package_dir,
    resolve_calibration,
)
from bielik_one.api.errors import INVALID_REQUEST, error_response
from bielik_one.api.http_guard import add_guard_arguments, guarded
from bielik_one.api.reasoning import EndpointReasoner


class MappedReadClient(httpx.Client):
    def __init__(self, *args, output_id_map, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = {int(k): int(v) for k, v in output_id_map.items()}

    def post(self, url, *, json=None, **kwargs):
        ids = json["logprob_token_ids"]
        mapped = [self.mapping[i] for i in ids]
        response = super().post(
            url, json={**json, "logprob_token_ids": mapped}, **kwargs
        )
        if not response.is_success:
            return response
        payload = response.json()
        for choice in payload["choices"]:
            tables = choice["logprobs"]["top_logprobs"]
            for pos, table in enumerate(tables):
                # Keep only explicitly requested entries. A separately returned
                # sampled native token could otherwise collide with a mapped ID.
                tables[pos] = {
                    f"token_id:{old}": table[f"token_id:{new}"]
                    for old, new in zip(ids, mapped)
                }
        return httpx.Response(
            response.status_code, json=payload, request=response.request
        )


def chat_payload(payload, blocked):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    existing = payload.get("bad_words") or []
    if not isinstance(existing, list) or any(not isinstance(x, str) for x in existing):
        raise ValueError("bad_words must be a list of strings")
    return {**payload, "bad_words": list(dict.fromkeys([*existing, *blocked]))}


class MaskChatTokens:
    def __init__(self, app, blocked):
        self.app = app
        self.blocked = blocked

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in [
            "/v1/chat/completions",
            "/v1/completions",
        ]:
            return await self.app(scope, receive, send)
        parts = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            body = json.dumps(
                chat_payload(json.loads(b"".join(parts)), self.blocked)
            ).encode()
        except (ValueError, TypeError) as exc:
            return await error_response(
                400, INVALID_REQUEST, str(exc), path=scope.get("path")
            )(scope, receive, send)
        modified = {
            **scope,
            "headers": [
                (k, v) for k, v in scope["headers"] if k.lower() != b"content-length"
            ]
            + [(b"content-length", str(len(body)).encode())],
        }
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(modified, replay, send)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--backend-url", default="http://127.0.0.1:8013")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--base-model", default="qwen-general")
    add_calibration_arguments(p)
    add_guard_arguments(p)
    add_limit_arguments(p)
    a = p.parse_args()
    meta = json.loads(a.metadata.read_text())
    meta["base_snapshot"] = str(package_dir(a.metadata, "tokenizer_config.json"))
    table, source = resolve_calibration(
        a.metadata, a.calibration, a.no_calibration, meta.get("source_repo")
    )
    print(f"calibration: {source or 'none (temperature 1)'}", flush=True)
    client = MappedReadClient(
        output_id_map=meta["output_id_map"],
        timeout=180,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=a.workers, max_keepalive_connections=a.workers
        ),
    )
    engine = RemoteVllmEngine(
        meta,
        a.backend_url,
        client=client,
        workers=a.workers,
        calibration=table,
        calibration_source=source,
        max_request_tokens=a.max_request_tokens,
        request_timeout=a.request_timeout,
    )
    # Reasoning must use the same chat-token mask as ordinary chat, not the
    # unmasked raw backend. FastAPI evaluation runs outside the event loop.
    from bielik_one.launch import connect_host

    own = connect_host(a.host)
    engine.reasoners["original"] = EndpointReasoner(
        f"http://{own}:{a.port}/v1", a.base_model, tok=getattr(engine, "tok", None)
    )
    app = MaskChatTokens(
        build_remote_app(engine, chat_model=a.base_model, error_format=a.error_format),
        meta["chat_blocked_tokens"],
    )
    uvicorn.run(guarded(app, a.max_body_mb, a.error_format), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
