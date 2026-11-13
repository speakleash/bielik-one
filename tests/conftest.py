"""Shared lightweight HTTP engine factories; no model weights or GPU required."""
import json
from types import SimpleNamespace

import httpx
import pytest

from bielik_one.api.remote_vllm import RemoteVllmEngine


@pytest.fixture
def engine_factory():
    engines = []

    def create(max_model_len=32768):
        requests = []
        anchor = 1500
        metadata = dict(
            choice_capacity=255,
            labels_capacity=248,
            label_ids=list(range(1000, 1248)),
            anchor_id=anchor,
            bias=[0.0] * 248,
            threshold=0.5,
            source_repo="test/decision",
        )

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            values = {f"token_id:{i}": -10.0 for i in body["logprob_token_ids"]}
            return httpx.Response(200, json={
                "usage": {"prompt_tokens": len(body["prompt"])},
                "choices": [{"logprobs": {"top_logprobs": [values]}}],
            })

        engine = RemoteVllmEngine(
            metadata,
            "http://backend",
            max_model_len=max_model_len,
            tokenizer=object(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        engines.append(engine)
        return SimpleNamespace(engine=engine, requests=requests)

    yield create
    for engine in engines:
        engine.close()
