"""Check a deployed `vllm serve --middleware bielik_one.api.vllm_middleware.SystemOneMiddleware` server.

    python scripts/check_middleware.py --url http://127.0.0.1:8000 --api-key "$VLLM_API_KEY" --model bielik-base

It sends a few requests and prints PASS/FAIL per check; the exit status is 0 only when every check passes:
- decisions and their readiness check answer (and need the API key when one is set);
- the internal LoRA name is hidden from /v1/models and refused in chat, completions and /v1/systemone;
- client backend flags and prompt logprobs are refused; ordinary and streaming chat still work;
- SageMaker adapter headers and runtime LoRA management are not usable by clients.

It generates a few tokens and runs one decision request; it changes nothing on the server.
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
    client = httpx.Client(base_url=args.url.rstrip("/"), timeout=600, trust_env=False, verify=verify)
    key = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail and not ok else ""))

    def post(path, body=None, headers=None, **kwargs):
        return client.post(path, json=body, headers={**key, **(headers or {})}, **kwargs)

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
        check("decisions need the API key", client.post("/v1/systemone", json=decision).status_code == 401)
        check("decision health needs the API key", client.get("/v1/systemone/health").status_code == 401)
    response = client.get("/v1/systemone/health", headers=key)
    check("decision health is ok", response.status_code == 200, response.text[:300])
    if response.status_code == 200:
        print("  health:", response.text)
    response = post("/v1/systemone", decision)
    check("decision answered", response.status_code == 200, response.text[:300])
    if response.status_code == 200:
        answer = response.json()["answers"]["sentiment"]
        print("  sentiment:", json.dumps(answer.get("probabilities")))
        check("negative review read as negative", answer.get("choice") == "neg", json.dumps(answer)[:300])
    response = post("/v1/systemone", {**decision, "model": args.lora_name})
    check("LoRA name refused at /v1/systemone", response.status_code == 422, response.text[:300])

    response = client.get("/v1/models", headers=key)
    ids = [card.get("id") for card in response.json().get("data", [])] if response.status_code == 200 else []
    check("LoRA hidden from /v1/models", args.model in ids and args.lora_name not in ids, f"{response.status_code} {ids}")

    chat = {"model": args.model, "messages": [{"role": "user", "content": "Odpowiedz jednym słowem: tak."}],
            "max_tokens": 4}
    response = post("/v1/chat/completions", chat)
    check("base model chat works", response.status_code == 200, response.text[:300])
    with client.stream("POST", "/v1/chat/completions", json={**chat, "stream": True}, headers=key) as stream:
        body = b"".join(stream.iter_raw())
    check("streaming chat works", stream.status_code == 200 and b"[DONE]" in body, body[:200].decode(errors="replace"))
    response = post("/v1/chat/completions", {**chat, "model": args.lora_name})
    check("LoRA name refused in chat", response.status_code == 400, response.text[:300])
    completion = {"model": args.lora_name, "prompt": "x", "max_tokens": 1}
    response = client.post("/v1/completions", content=json.dumps(completion), headers={**key, "content-type": "text/plain"})
    check("LoRA name refused whatever the content type", response.status_code == 400, response.text[:300])
    response = post("/v1/completions", {"model": args.model, "prompt": "x", "max_tokens": 1, "prompt_logprobs": 1})
    check("prompt logprobs refused", response.status_code == 400, response.text[:300])

    invocation = {"model": args.model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    response = post("/invocations", invocation, {"X-Amzn-SageMaker-Adapter-Identifier": args.lora_name})
    check("SageMaker adapter header refused", response.status_code == 400, response.text[:300])
    for method, path, body in [
        ("POST", "/v1/unload_lora_adapter", {"lora_name": args.lora_name}),
        ("POST", "/adapters", {"name": "copy", "src": "/nonexistent"}),
        ("DELETE", f"/adapters/{args.lora_name}", None),
    ]:
        response = client.request(method, path, json=body, headers=key)
        # 404/405: vLLM registers these only with VLLM_ALLOW_RUNTIME_LORA_UPDATING (refused by the heads plugin).
        check(f"{method} {path} not usable by clients", response.status_code in (403, 404, 405),
              f"{response.status_code} {response.text[:200]}")
    response = client.get("/v1/systemone/health", headers=key)
    check("decision adapter still loaded", response.status_code == 200, response.text[:300])

    if client.post("/invocations", json=invocation).status_code != 401:
        print("  note: /invocations answers without the API key (vLLM guards only /v1 paths); restrict it in a proxy")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
