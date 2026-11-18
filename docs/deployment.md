# Exposing bielik-one on a network

`bielik-one frontend` (and `vllm serve` with the SystemOne middleware) has no authentication and no TLS of its own. The raw vLLM backend accepts internal decision flags. Anything that can reach these ports can use the model and the GPU. This page describes the layout we recommend when other machines need access, and how to check it.

## Layout

```
client ──TLS──> reverse proxy :443 ──> bielik-one frontend 127.0.0.1:8026 ──> vLLM backend 127.0.0.1:8025
               (authentication,        (typed decisions, chat proxy,           (never reachable from outside)
                limits, route list)     request limits)
```

- Keep the frontend on loopback (the default `--host 127.0.0.1`) and the backend on loopback (the default `--backend-host 127.0.0.1`). If the frontend must listen on another interface, firewall it so that only the proxy can connect. The frontend prints a warning at startup when it listens beyond loopback.
- With the middleware, `vllm serve --host 127.0.0.1` takes the place of both, and the proxy points at its port.

Reference nginx configurations:

- [`deploy/nginx/bielik-one.conf`](../deploy/nginx/bielik-one.conf) for `bielik-one frontend`;
- [`deploy/nginx/vllm-middleware.conf`](../deploy/nginx/vllm-middleware.conf) for `vllm serve` with the middleware.

## What the proxy does

| Concern | In the reference configuration |
|---|---|
| TLS | terminated at the proxy, TLS 1.2 and 1.3, port 80 redirects to 443 |
| Authentication | every route except the load-balancer probe; credentials are not forwarded upstream |
| Routes | only `POST /v1/systemone`, `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`; everything else is 404, including `/docs`, `/openapi.json` and, with the middleware, vLLM's `/metrics`, `/invocations`, `/tokenize`, scoring and runtime administration routes |
| Probe | `GET /health` (frontend) or `GET /v1/systemone/health` (middleware) without credentials, only from the addresses you allow |
| Body size | 16 MiB, the same as the frontend's `--max-body-mb` default, refused before it is read |
| Slow clients | header and body timeouts, so they hold the proxy instead of the frontend |
| Rate | 10 requests/s per client address with a burst of 20, and 16 open connections per address; over that, 429 |
| Request ids | `$request_id` is sent as `X-Request-Id`, so one id appears in the proxy log and in the frontend's log |
| Internal header | with the middleware, `X-SystemOne-Internal` is removed from client requests |

Tune the rate and connection limits to your traffic. They protect the GPU from one client, not from many; the per-request limits of the frontend (body size, decision reads, generation budget) still apply behind them.

## Authentication options

Do not build your own scheme. Use one your organisation already runs:

- **HTTP Basic over TLS** (as in the reference files): create the file with the first user, `htpasswd -cB /etc/nginx/bielik-one.htpasswd alice`, and add later users without `-c` (`htpasswd -B /etc/nginx/bielik-one.htpasswd bob`), because `-c` overwrites an existing file. If `htpasswd` is not installed, append one `user:hash` line per user: `printf 'alice:%s\n' "$(openssl passwd -apr1)" | sudo tee -a /etc/nginx/bielik-one.htpasswd`. Suitable for a few service accounts.
- **Single sign-on (OIDC):** put an authenticating proxy such as oauth2-proxy next to nginx and replace `auth_basic` with `auth_request /oauth2/auth;` following that proxy's nginx guide.
- **Client certificates (mTLS):** for machine clients, replace `auth_basic` with `ssl_client_certificate /etc/nginx/tls/clients-ca.pem; ssl_verify_client on;`.

With the middleware, vLLM's own `--api-key` covers only paths under `/v1`. The proxy authenticates every path. If you also keep the vLLM key, set it upstream with `proxy_set_header Authorization "Bearer <key>";` so clients never see it.

## Checklist

1. Frontend and vLLM listen on `127.0.0.1` (`ss -ltnp` shows no `0.0.0.0`/`*` for their ports), or a firewall admits only the proxy.
2. The package is pinned: `--revision <commit>` and `--manifest-sha256 <digest>` (see the README).
3. The proxy has a real certificate, and port 80 only redirects.
4. Without credentials, every route except the probe gives 401, and the probe gives 403 from outside the allowed networks.
5. With credentials, a decision and a short chat request succeed; `/docs` and any route not listed give 404.
6. A body over 16 MiB gives 413 at the proxy.

## Checking it

From a machine outside the allowed probe networks, with `$H` the proxy's host name:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://$H/v1/systemone -X POST -d '{}'            # 401
curl -s -o /dev/null -w '%{http_code}\n' https://$H/health                                  # 403
curl -s -o /dev/null -w '%{http_code}\n' -u alice https://$H/docs                           # 404
curl -s -u alice https://$H/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "decision", "state": "The invoice is overdue.",
  "questions": {"late": {"type": "noul", "instructions": "Is the payment late?"}}}'         # an answer (frontend)
curl -s -o /dev/null -w '%{http_code}\n' http://$H:8026/health                             # connection refused
```

With the middleware, `decision` is the internal adapter name and is refused (422). Use the base model's public name instead, one of those listed by `curl -s -u alice https://$H/v1/models` (for example `bielik-base`):

```bash
curl -s -u alice https://$H/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "bielik-base", "state": "The invoice is overdue.",
  "questions": {"late": {"type": "noul", "instructions": "Is the payment late?"}}}'         # an answer (middleware)
```
