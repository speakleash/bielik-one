"""Check a deployed `vllm serve --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware` server.

    python scripts/check_middleware.py --url http://127.0.0.1:8000 --api-key "$VLLM_API_KEY" --model bielik-base

It sends a few requests and prints PASS/FAIL per check; the exit status is 0 only when every check passes:
- decisions and their readiness check answer (and need the API key when one is set);
- the internal LoRA name is hidden from /v1/models and refused (by the middleware itself) in chat and completions;
- client backend flags and prompt logprobs are refused; ordinary and streaming chat still work;
- SageMaker adapter headers and runtime LoRA management are not usable by clients.

It generates a few tokens and runs one decision request; it changes nothing on the server. A negative sample review
read as anything else is printed as a warning (a model sanity hint), not counted as a failure.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys

import httpx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the vLLM server (default: %(default)s)")
    parser.add_argument("--api-key", help="vLLM's API key, when the server has one")
    parser.add_argument("--model", required=True, help="the base model's served name (--served-model-name)")
    parser.add_argument("--lora-name", default="decision", help="the internal LoRA name (default: %(default)s)")
    parser.add_argument("--cacert", help="CA bundle (PEM) for an https server with a private certificate")
    args = parser.parse_args(argv)

    verify = ssl.create_default_context(cafile=args.cacert) if args.cacert else True
    client = httpx.Client(base_url=args.url.rstrip("/"), timeout=httpx.Timeout(600, connect=10), trust_env=False,
                          verify=verify)
    key = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    results: list[bool] = []
    refused = "answers typed decisions"  # the middleware's own refusal of the LoRA name (not vLLM's)

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail and not ok else ""))

    def call(method, path, body=None, headers=None, content=None):
        """The response, or None (reported) when the server cannot be reached."""
        try:
            return client.request(method, path, json=body if content is None else None, content=content,
                                  headers={**key, **(headers or {})})
        except httpx.HTTPError as exc:
            print(f"FAIL {method} {path} -- {exc!r}")
            results.append(False)
            return None

    def status(response):
        return None if response is None else response.status_code

    def text(response):
        return "" if response is None else response.text[:300]

    decision = {
        "model": args.model,
        "state": "Recenzja: Film był nudny, przewidywalny i zdecydowanie za długi.",
        "questions": {
            "sentiment": {"type": "choice", "instructions": "Jaki jest wydźwięk recenzji?",
                          "criteria": {"pos": "pozytywny", "neg": "negatywny"}},
            "spoiler": {"type": "noul", "instructions": "Czy recenzja zdradza zakończenie filmu?"},
        },
    }

    if args.api_key:
        try:
            check("decisions need the API key", client.post("/v1/systemone", json=decision).status_code == 401)
            check("decision health needs the API key", client.get("/v1/systemone/health").status_code == 401)
        except httpx.HTTPError as exc:
            check("server reachable", False, repr(exc))
            return 1
    response = call("GET", "/v1/systemone/health")
    check("decision health is ok", status(response) == 200, text(response))
    if status(response) == 200:
        print("  health:", response.text)
    response = call("POST", "/v1/systemone", decision)
    check("decision answered", status(response) == 200, text(response))
    if status(response) == 200:
        answer = response.json().get("answers", {}).get("sentiment", {})
        print("  sentiment:", json.dumps(answer.get("probabilities")))
        if answer.get("choice") != "neg":  # a model sanity hint, not a deployment check
            print("  warning: the negative sample review was not read as negative; check the package and base model")

    response = call("GET", "/v1/models")
    ids = [card.get("id") for card in response.json().get("data", [])] if status(response) == 200 else []
    check("LoRA hidden from /v1/models", args.model in ids and args.lora_name not in ids, f"{status(response)} {ids}")

    chat = {"model": args.model, "messages": [{"role": "user", "content": "Odpowiedz jednym słowem: tak."}],
            "max_tokens": 4}
    response = call("POST", "/v1/chat/completions", chat)
    check("base model chat works", status(response) == 200, text(response))
    try:
        with client.stream("POST", "/v1/chat/completions", json={**chat, "stream": True}, headers=key) as stream:
            body = b"".join(stream.iter_raw())
        check("streaming chat works", stream.status_code == 200 and b"[DONE]" in body,
              body[:200].decode(errors="replace"))
    except httpx.HTTPError as exc:
        check("streaming chat works", False, repr(exc))
    # The refusal text proves the middleware (and the --lora-name given here) answered, not vLLM's own validation.
    response = call("POST", "/v1/chat/completions", {**chat, "model": args.lora_name})
    check("LoRA name refused in chat", status(response) == 400 and refused in text(response), text(response))
    completion = json.dumps({"model": args.lora_name, "prompt": "x", "max_tokens": 1})
    response = call("POST", "/v1/completions", content=completion, headers={"content-type": "text/plain"})
    check("LoRA name refused whatever the content type", status(response) == 400 and refused in text(response),
          text(response))
    response = call("POST", "/v1/completions", {"model": args.model, "prompt": "x", "max_tokens": 1,
                                                "prompt_logprobs": 1})
    check("prompt logprobs refused", status(response) == 400 and "prompt logprobs" in text(response), text(response))

    invocation = {"model": args.model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    response = call("POST", "/invocations", invocation, {"X-Amzn-SageMaker-Adapter-Identifier": args.lora_name})
    check("SageMaker adapter header refused", status(response) == 400 and "SageMaker" in text(response),
          text(response))
    # Invalid bodies: a guarded route answers 403, an absent one 404/405, an unguarded one 400/422 without
    # unloading anything (VLLM_ALLOW_RUNTIME_LORA_UPDATING registers them; the heads plugin refuses that flag).
    for path in ("/v1/unload_lora_adapter", "/v1/load_lora_adapter", "/adapters"):
        response = call("POST", path, {})
        check(f"POST {path} not usable by clients", status(response) in (403, 404, 405),
              f"{status(response)} {text(response)}")
    response = call("GET", "/v1/systemone/health")
    check("decision adapter still loaded", status(response) == 200, text(response))

    response = call("POST", "/invocations", invocation, headers={"Authorization": ""})
    if status(response) not in (None, 401):
        print("  note: /invocations answers without the API key (vLLM guards only /v1 paths); restrict it in a proxy")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
