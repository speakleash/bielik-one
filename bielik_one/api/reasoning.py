"""Optional OpenAI-compatible reasoning endpoints, configured by the server operator."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from bielik_one.prompting.inference import reasoning_messages


# Reasoning endpoints are local or operator-configured; never route them through HTTP_PROXY.
_OPENER = build_opener(ProxyHandler({}))


class ReasoningError(RuntimeError):
    """The reasoning endpoint failed; `status` is 422 when it rejected the request (e.g. context too long), 504 when
    it timed out, 502 otherwise."""

    def __init__(self, msg: str, status: int = 502):
        super().__init__(msg)
        self.status = status


class EndpointReasoner:
    def __init__(
        self, url: str, model: str, key_env: str = "REASONING_API_KEY", tok=None
    ):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key_env = key_env
        self.tok = tok  # neutralizes client text against this tokenizer's control tokens

    def __call__(self, row, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": reasoning_messages(row, self.tok),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        # A non-streaming generation sends nothing until it ends: allow for slow decoding of long analyses.
        timeout = max(180, 60 + max_tokens / 10)
        try:
            with _OPENER.open(
                Request(self.url, data=json.dumps(payload).encode(), headers=headers),
                timeout=timeout,
            ) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace")
            raise ReasoningError(
                f"reasoning endpoint returned {exc.code}: {detail}",
                # Only a request the endpoint found too long or malformed is the client's to change; an auth or
                # rate-limit refusal is the server's configuration or load.
                422 if exc.code in (400, 413, 422) else 502,
            ) from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
            raise ReasoningError(f"reasoning endpoint failed: {exc}", 504 if timed_out else 502) from exc
        choice = result["choices"][0]
        message = choice["message"]
        text = "\n".join(
            value
            for value in (message.get("reasoning_content"), message.get("content"))
            if value
        )
        if not text.strip():
            raise ReasoningError("reasoning endpoint returned an empty analysis")
        usage = result.get("usage", {})
        return {
            "text": text,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "truncated": choice.get("finish_reason") == "length",
        }
