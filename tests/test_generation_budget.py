"""Per-request budget for public generation: n and the output tokens one chat/completions request may ask for."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bielik_one import launch
from bielik_one.api import remote_vllm
from bielik_one.api import vllm_middleware as middleware
from bielik_one.api.remote_vllm import GenerationLimits, check_generation_budget, public_request_body
from bielik_one.api.server import RequestError

from tests.test_launch import dry, package
from tests.test_vllm_middleware import make_app

LIMITS = GenerationLimits(max_n=16, max_tokens=131_072, context_tokens=32_768)


CHAT, COMPLETIONS, BATCH, RESPONSES = "/v1/chat/completions", "/v1/completions", "/v1/chat/completions/batch", "/v1/responses"


def refused(body, route=None, limits=LIMITS):
    with pytest.raises(RequestError) as caught:
        check_generation_budget(body, limits, route)
    return caught.value


# -- what is counted ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, body",
    [
        (CHAT, {"messages": [{"role": "user", "content": "hi"}]}),  # one answer, up to the full context
        (CHAT, {"messages": [], "n": 4}),  # four full-context answers: exactly the budget
        (None, {"messages": [], "n": 4}),  # the same on a route the filter does not know
        (CHAT, {"messages": [], "n": 16, "max_tokens": 8192}),
        (CHAT, {"messages": [], "n": 16, "max_completion_tokens": 8192}),
        (COMPLETIONS, {"prompt": ["a"] * 1024, "max_tokens": 128}),  # 1024 prompts x 128 tokens
        (COMPLETIONS, {"prompt": list(range(5000)), "n": 4}),  # one prompt given as token ids, not 5000 prompts
        (BATCH, {"messages": [[{"role": "user", "content": "a"}]] * 8, "max_tokens": 16384}),
        (RESPONSES, {"input": "hi", "max_output_tokens": 4096, "n": 16}),
        (CHAT, {"messages": [], "max_tokens": 10**9}),  # longer than the context counts as the context
        ("/api" + CHAT, {"messages": [], "n": 16, "max_tokens": 8192}),  # behind a root path
    ],
)
def test_requests_within_the_budget_pass(route, body):
    check_generation_budget(body, LIMITS, route)


def test_n_over_the_limit_is_refused():
    error = refused({"messages": [], "n": 1000, "max_tokens": 1}, CHAT)
    assert error.loc == ["body", "n"]
    assert "n is 1000; this server allows at most 16" in error.msg


def test_best_of_and_beam_width_are_counted():
    error = refused({"prompt": "x", "n": 1, "best_of": 64, "max_tokens": 1}, COMPLETIONS)
    assert error.loc == ["body", "best_of"]
    error = refused({"prompt": "x", "n": 17, "use_beam_search": True, "max_tokens": 1}, COMPLETIONS)  # beam width
    assert error.loc == ["body", "n"]


@pytest.mark.parametrize(
    "route, body, field, total",
    [
        (CHAT, {"messages": [], "n": 5}, "max_tokens", 5 * 32_768),  # not set: full context per answer
        (CHAT, {"messages": [], "n": 16, "max_tokens": 16_384}, "max_tokens", 16 * 16_384),
        (CHAT, {"messages": [], "max_completion_tokens": 32_768, "n": 8}, "max_completion_tokens", 8 * 32_768),
        (COMPLETIONS, {"prompt": ["a"] * 1024, "max_tokens": 256}, "max_tokens", 1024 * 256),  # many prompts
        (COMPLETIONS, {"prompt": [[1, 2]] * 64, "max_tokens": 4096}, "max_tokens", 64 * 4096),  # token-id prompts
        (BATCH, {"messages": [[{"role": "user", "content": "a"}]] * 64, "max_tokens": 4096}, "max_tokens", 64 * 4096),
        (RESPONSES, {"input": "hi", "max_output_tokens": 32_768, "n": 5}, "max_output_tokens", 5 * 32_768),
    ],
)
def test_requests_over_the_token_budget_are_refused(route, body, field, total):
    error = refused(body, route)
    assert error.loc == ["body", field]
    assert f"up to {total} output tokens" in error.msg and "at most 131072" in error.msg


def test_unset_max_tokens_is_named_in_the_message():
    assert "(max_tokens not set)" in refused({"messages": [], "n": 8}, CHAT).msg


@pytest.mark.parametrize("n", ["1000", " 1000 ", 1000.0, "1e3", 16.5])
def test_counts_are_read_the_way_vllm_coerces_them(n):
    # vLLM's pydantic models accept "1000" and 1000.0 as n=1000; a fraction is rounded up, never down.
    assert refused({"messages": [], "n": n, "max_tokens": 1}, CHAT).loc == ["body", "n"]


@pytest.mark.parametrize("max_tokens", ["32768", 32768.0])
def test_token_limits_are_read_the_way_vllm_coerces_them(max_tokens):
    assert refused({"messages": [], "n": 5, "max_tokens": max_tokens}, CHAT).loc == ["body", "max_tokens"]


def test_values_vllm_would_reject_anyway_are_not_counted():
    for body in ({"messages": [], "n": "many"}, {"messages": [], "n": None}, {"messages": [], "max_tokens": [5]},
                 {"messages": [], "n": float("nan")}, {"messages": [], "n": True}):
        check_generation_budget(body, LIMITS, CHAT)


def test_only_the_field_the_route_reads_counts():
    # Chat ignores max_output_tokens, so it cannot stand in for an unset max_tokens.
    error = refused({"messages": [], "n": 16, "max_output_tokens": 1}, CHAT)
    assert error.loc == ["body", "max_tokens"] and "up to 524288 output tokens" in error.msg
    # Completions reads max_tokens; a small max_completion_tokens next to it must not hide it.
    error = refused({"prompt": "x", "n": 8, "max_completion_tokens": 1, "max_tokens": 32_768}, COMPLETIONS)
    assert error.loc == ["body", "max_tokens"] and "up to 262144 output tokens" in error.msg
    # Chat prefers max_completion_tokens, as vLLM does, so a huge max_tokens next to it does not count.
    check_generation_budget({"messages": [], "n": 16, "max_completion_tokens": 8192, "max_tokens": 10**9}, LIMITS, CHAT)


def test_routes_the_filter_does_not_know_count_the_costliest_reading():
    # /invocations hands the body to the chat or the completions handler, whichever validates.
    body = {"messages": [], "n": 16, "max_completion_tokens": 8192, "max_tokens": 10**9}
    assert refused(body, "/invocations").loc == ["body", "max_tokens"]
    assert refused({"messages": [], "n": 16, "max_output_tokens": 1}, "/invocations").loc == ["body", "max_tokens"]


def test_messages_cannot_lower_the_prompt_count_of_completions():
    # vLLM ignores messages on /v1/completions and answers all 1024 prompts.
    body = {"prompt": ["x"] * 1024, "messages": [[]], "max_tokens": 256}
    error = refused(body, COMPLETIONS)
    assert "up to 262144 output tokens (1024 prompt(s)" in error.msg
    # And the other way round on batch chat.
    body = {"prompt": "x", "messages": [[{"role": "user", "content": "a"}]] * 64, "max_tokens": 4096}
    assert "(64 prompt(s)" in refused(body, BATCH).msg


def test_token_in_sampling_params_are_checked():
    check_generation_budget({"token_ids": [1, 2], "sampling_params": {"n": 2, "max_tokens": 64}}, LIMITS)
    assert refused({"token_ids": [1], "sampling_params": {"n": 1000}}).loc == ["body", "sampling_params", "n"]
    error = refused({"token_ids": [1], "sampling_params": {"n": 8, "max_tokens": "32768"}})
    assert error.loc == ["body", "sampling_params", "max_tokens"]


def test_limits_can_be_lifted():
    check_generation_budget({"prompt": ["a"] * 1024, "n": 16384}, GenerationLimits(max_n=None, max_tokens=None))


# -- public_request_body: the shared filter of the frontend and the middleware ---------------------------------------


def test_public_request_body_applies_the_budget_and_keeps_valid_bodies_as_sent():
    raw = json.dumps({"model": "chat", "messages": [], "n": 2, "max_tokens": 100}).encode()
    assert public_request_body(raw, {"decision"}, "chat") is raw
    with pytest.raises(RequestError):
        public_request_body(json.dumps({"model": "chat", "messages": [], "n": 1000}).encode(), {"decision"}, "chat")
    lifted = GenerationLimits(max_n=None, max_tokens=None)
    body = json.dumps({"model": "chat", "messages": [], "n": 1000}).encode()
    assert public_request_body(body, {"decision"}, "chat", lifted) is body


TIGHT = GenerationLimits(max_n=2, max_tokens=4096, context_tokens=32_768)


@pytest.mark.parametrize(
    "route, body",
    [
        ("/tokenize", {"model": "chat", "prompt": "hello"}),
        ("/tokenize", {"model": "chat", "messages": [{"role": "user", "content": "hi"}]}),
        ("/detokenize", {"model": "chat", "tokens": [1, 2, 3]}),
        ("/v1/messages/count_tokens", {"model": "chat", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/chat/completions/render", {"model": "chat", "messages": []}),
        ("/v1/completions/render", {"model": "chat", "prompt": "hello"}),
        ("/generative_scoring", {"model": "chat", "query": "q", "items": ["a", "b"]}),
        ("/v1/embeddings", {"model": "chat", "input": "hello"}),
        ("/api/tokenize", {"model": "chat", "prompt": "hello"}),  # behind a root path
        ("/pause", {}),  # an admin call with no generation fields
    ],
)
def test_routes_that_do_not_generate_are_not_charged(route, body):
    # A budget below the context would refuse each of these if they were counted as full-context answers.
    raw = json.dumps(body).encode()
    assert public_request_body(raw, {"decision"}, None, TIGHT, route) is raw


@pytest.mark.parametrize(
    "route, body",
    [
        (CHAT, {"model": "chat", "messages": []}),  # one full-context answer is over 4096
        (COMPLETIONS, {"model": "chat", "prompt": "hello", "max_tokens": 5000}),
        ("/invocations", {"model": "chat", "prompt": "hello"}),  # unknown route, generation body
        ("/inference/v1/generate", {"model": "chat", "token_ids": [1], "sampling_params": {"max_tokens": 5000}}),
        ("/v1/chat/completions/batch", {"model": "chat", "messages": [[{"role": "user", "content": "a"}]] * 3}),
    ],
)
def test_routes_that_generate_are_still_charged(route, body):
    with pytest.raises(RequestError):
        public_request_body(json.dumps(body).encode(), {"decision"}, None, TIGHT, route)


# -- separate frontend ----------------------------------------------------------------------------------------------


class Engine:
    backend_url = "http://127.0.0.1:9"  # nothing listens: an accepted request ends as 502
    aliases = {"decision"}
    temperatures = {}
    calibration_source = None
    max_model_len = 32_768

    def close(self):
        pass


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_frontend_refuses_over_budget_requests_before_vllm(path):
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat")
    with TestClient(app) as client:
        response = client.post(path, json={"model": "chat", "prompt": "x", "messages": [], "n": 1000})
        assert response.status_code == 400
        assert "n is 1000" in response.json()["error"]["message"]  # chat routes answer in the OpenAI format
        response = client.post(path, json={"model": "chat", "prompt": "x", "messages": [], "n": 2, "max_tokens": 64})
        assert response.status_code == 502  # passed the filter, then found no vLLM


def test_frontend_passes_the_route():
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat")
    with TestClient(app) as client:
        body = {"model": "chat", "prompt": ["x"] * 1024, "messages": [[]], "max_tokens": 256}
        assert client.post("/v1/completions", json=body).status_code == 400  # 1024 prompts, not 1 conversation
        assert client.post("/v1/chat/completions", json={"messages": [], "n": 16, "max_output_tokens": 1}).status_code == 400


def test_frontend_uses_configured_limits():
    limits = GenerationLimits(max_n=2, max_tokens=1000)
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat", generation_limits=limits)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"messages": [], "n": 3, "max_tokens": 1}).status_code == 400
        too_long = client.post("/v1/chat/completions", json={"messages": [], "n": 2, "max_tokens": 501})
        assert too_long.status_code == 400 and "at most 1000" in too_long.json()["error"]["message"]


# -- vLLM middleware ------------------------------------------------------------------------------------------------


def test_middleware_refuses_over_budget_requests_on_every_public_route():
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/invocations"):
            response = client.post(path, json={"model": "base-a", "messages": [], "n": 1000})
            assert response.status_code == 400, path
            assert "n is 1000" in response.json()["error"]["message"]
            assert client.post(path, json={"model": "base-a", "messages": [], "n": "1000"}).status_code == 400, path
            ok = client.post(path, json={"model": "base-a", "messages": [], "n": 2, "max_tokens": 64})
            assert ok.status_code == 200, path
        # The route decides which limit field counts: chat ignores max_output_tokens, responses reads it.
        over = {"model": "base-a", "messages": [], "input": "x", "n": 16, "max_output_tokens": 1}
        assert client.post("/v1/chat/completions", json=over).status_code == 400
        assert client.post("/v1/responses", json=over).status_code == 200
        # Token-in requests carry n and max_tokens in sampling_params.
        token_in = {"model": "base-a", "token_ids": [1, 2], "sampling_params": {"n": 1000, "max_tokens": 1}}
        assert client.post("/inference/v1/generate", json=token_in).status_code == 400
        token_in["sampling_params"]["n"] = 2
        assert client.post("/inference/v1/generate", json=token_in).status_code == 200


def test_middleware_settings_and_internal_reads():
    app, owner, _, _ = make_app(max_generation_n=2, max_generation_tokens=None)
    with TestClient(app) as client:
        assert client.post("/v1/completions", json={"model": "base-a", "prompt": "x", "n": 3}).status_code == 400
        assert client.post("/v1/completions", json={"model": "base-a", "prompt": ["x"] * 1024}).status_code == 200
        # The middleware's own decision reads are not public requests and are not filtered.
        internal = {"x-systemone-internal": owner.settings.internal_token}
        assert client.post("/v1/completions", json={"model": "decision", "prompt": "x", "n": 3}, headers=internal).status_code == 200


def test_middleware_settings_from_env():
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package"})
    assert (settings.max_generation_n, settings.max_generation_tokens) == (
        remote_vllm.DEFAULT_MAX_GENERATION_N, remote_vllm.DEFAULT_MAX_GENERATION_TOKENS)
    settings = middleware.Settings.from_env(
        {"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MAX_GENERATION_N": "4", "SYSTEMONE_MAX_GENERATION_TOKENS": "none"})
    assert (settings.max_generation_n, settings.max_generation_tokens) == (4, None)


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
@pytest.mark.parametrize("name", ["SYSTEMONE_MAX_GENERATION_N", "SYSTEMONE_MAX_GENERATION_TOKENS"])
def test_middleware_rejects_invalid_limits(name, value):
    with pytest.raises(RuntimeError, match=name):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", name: value})


# -- launcher and frontend command line -----------------------------------------------------------------------------


def test_launcher_passes_limits_to_the_frontend_only_when_given():
    for kind in launch.PORTS:
        with tempfile.TemporaryDirectory() as d:
            package(kind, Path(d))
            command = dry("frontend", d)["command"]
            assert "--max-generation-n" not in command and "--max-generation-tokens" not in command
            command = dry("frontend", d, "--max-generation-n", "4", "--max-generation-tokens", "none")["command"]
            assert command[command.index("--max-generation-n") + 1] == "4"
            assert command[command.index("--max-generation-tokens") + 1] == "none"


@pytest.mark.parametrize("args", [["--max-generation-n", "0"], ["--max-generation-tokens", "-5"], ["--max-generation-n", "x"]])
def test_launcher_rejects_invalid_limits(args):
    with tempfile.TemporaryDirectory() as d:
        package("qwen-merged", Path(d))
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            launch.main(["frontend", d, *args, "--dry-run"])


def test_frontend_parser_defaults_and_none():
    import argparse

    parser = argparse.ArgumentParser()
    remote_vllm.add_generation_arguments(parser)
    defaults = parser.parse_args([])
    assert remote_vllm.generation_limits(defaults) == GenerationLimits()
    parsed = parser.parse_args(["--max-generation-n", "none", "--max-generation-tokens", "5000"])
    assert remote_vllm.generation_limits(parsed, 4096) == GenerationLimits(None, 5000, 4096)
    for args in (["--max-generation-n", "0"], ["--max-generation-tokens", "abc"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            parser.parse_args(args)


# -- review fixes ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [{"n": -1}, {"n": 0}, {"n": -4, "best_of": -8}, {"best_of": -1}])
def test_negative_or_zero_counts_do_not_lower_the_cost(extra):
    # /v1/responses ignores n, so vLLM still answers once with the full max_output_tokens.
    tight = GenerationLimits(max_n=16, max_tokens=4096, context_tokens=32_768)
    error = refused({"input": "x", "max_output_tokens": 8192, **extra}, RESPONSES, tight)
    assert "1 answer(s) x 8192 tokens" in error.msg


def test_token_in_route_reads_only_sampling_params():
    tight = GenerationLimits(max_n=16, max_tokens=4096, context_tokens=32_768)
    route = "/inference/v1/generate"
    check_generation_budget({"token_ids": [1], "sampling_params": {"max_tokens": 32}}, tight, route)
    error = refused({"token_ids": [1], "sampling_params": {"n": 2, "max_tokens": 4096}}, route, tight)
    assert error.loc == ["body", "sampling_params", "max_tokens"]
    # Without sampling_params the request is still charged as before.
    refused({"token_ids": [1]}, route, tight)


def test_middleware_charges_unset_limits_with_the_served_context():
    from types import SimpleNamespace

    # The budget allows one 40,000-token answer; vLLM serves a 65,536-token context, so an answer without max_tokens
    # may be longer than that, and an engine default of 32,768 would let it through.
    app, _, _, _ = make_app(max_generation_tokens=40_000)
    app.state.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=65_536))
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"model": "base-a", "messages": []}).status_code == 400
        ok = {"model": "base-a", "messages": [], "max_tokens": 40_000}
        assert client.post("/v1/chat/completions", json=ok).status_code == 200
