"""The reference proxy configurations stay in step with the code, and the frontend warns when it is exposed."""

import re
from pathlib import Path

import pytest

from bielik_one.api import http_guard, remote_vllm
from bielik_one.api import vllm_middleware as middleware

DEPLOY = Path(__file__).resolve().parent.parent / "deploy" / "nginx"
FRONTEND = (DEPLOY / "bielik-one.conf").read_text()
MIDDLEWARE = (DEPLOY / "vllm-middleware.conf").read_text()
PUBLIC = {"/v1/systemone": "POST", "/v1/chat/completions": "POST", "/v1/completions": "POST", "/v1/models": "GET"}


def locations(conf: str) -> dict[str, str]:
    """Exact-match locations and the method their limit_except allows."""
    out = {}
    for path, block in re.findall(r"location = (\S+) \{(.*?)\n    \}", conf, re.S):
        method = re.search(r"limit_except (\w+)", block)
        out[path] = method.group(1) if method else None
    return out


class Engine:
    backend_url = "http://127.0.0.1:9"
    aliases = {"decision"}
    temperatures = {}
    calibration_source = None
    max_model_len = 32768

    def close(self):
        pass


def test_frontend_proxy_forwards_exactly_the_public_routes_the_frontend_serves():
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat")
    served = {(route.path, method) for route in app.routes for method in (getattr(route, "methods", None) or ())}
    forwarded = locations(FRONTEND)
    assert forwarded == {**PUBLIC, "/health": "GET"}
    for path, method in forwarded.items():
        assert (path, method) in served, path


def test_both_proxies_authenticate_limit_and_close_everything_else():
    for conf in (FRONTEND, MIDDLEWARE):
        assert re.search(r"^\s*auth_basic\s+\"bielik-one\";", conf, re.M)
        assert re.search(r"location / \{\s*return 404;\s*\}", conf)
        assert "ssl_protocols       TLSv1.2 TLSv1.3;" in conf
        assert re.search(r"return 301 https://", conf)
        assert 'proxy_set_header   Authorization "";' in conf
        # The proxy refuses bodies at the frontend's own default limit, before reading them.
        assert f"client_max_body_size  {http_guard.DEFAULT_MAX_BODY_MB}m;" in conf
        # The probe is the only route without credentials, and only from allowed networks.
        unauthenticated = re.findall(r"location = (\S+) \{[^}]*?auth_basic off;", conf, re.S)
        assert len(unauthenticated) == 1
        probe = re.search(r"location = %s \{(.*?)\n    \}" % re.escape(unauthenticated[0]), conf, re.S).group(1)
        assert "deny  all;" in probe


def test_middleware_proxy_matches_the_middleware():
    forwarded = locations(MIDDLEWARE)
    assert forwarded == {**PUBLIC, middleware.HEALTH_PATH: "GET"}
    assert middleware.DECISION_PATH in forwarded
    # Clients must never be able to send the header of the middleware's own loopback reads.
    header = middleware.INTERNAL_HEADER
    assert re.search(r'proxy_set_header\s+%s ""' % re.escape(header), MIDDLEWARE, re.I)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_no_warning_on_loopback(host):
    assert http_guard.exposure_warning(host) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.5.10", "decisions.internal"])
def test_warning_beyond_loopback(host):
    warning = http_guard.exposure_warning(host)
    assert warning.startswith("WARNING:") and host in warning and "docs/deployment.md" in warning


def test_the_guide_links_files_that_exist():
    guide = (DEPLOY.parent.parent / "docs" / "deployment.md").read_text()
    for target in re.findall(r"\]\(\.\./([^)]+)\)", guide):
        assert (DEPLOY.parent.parent / target).is_file(), target
