"""Guarded HTTP GET for research tools (stdlib only).

- http/https only; no credentials in URLs
- refuses private, loopback, link-local and otherwise non-public addresses - including
  after redirects - so research cannot reach the developer's local network
- hard timeout and response-size cap
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from harness.tools.base import ToolError

USER_AGENT = "ai-coding-harness-researcher/0.1 (+documentation lookup)"
DEFAULT_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class HttpResponse:
    url: str
    final_url: str
    status: int
    content_type: str
    body: bytes

    def text(self) -> str:
        charset = "utf-8"
        if "charset=" in self.content_type:
            charset = self.content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        return self.body.decode(charset, errors="replace")


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...


def validate_public_url(url: str) -> str:
    """Return the normalized URL or raise ToolError if it is not a public http(s) URL."""
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"Only http(s) URLs are allowed, got {parsed.scheme or 'none'!r}")
    if not parsed.hostname:
        raise ToolError("URL has no host")
    if parsed.username or parsed.password:
        raise ToolError("URLs with embedded credentials are not allowed")
    host = parsed.hostname
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ToolError(f"Refusing to fetch a local host: {host}")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ToolError(f"Could not resolve host {host}: {exc}") from None
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ToolError(f"Refusing to fetch non-public address {address} ({host})")
    return urllib.parse.urlunsplit(parsed._replace(fragment=""))


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibHttpClient:
    def __init__(self, *, timeout: float = 15.0, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._opener = urllib.request.build_opener(_SafeRedirects())

    def get(self, url: str) -> HttpResponse:
        target = validate_public_url(url)
        request = urllib.request.Request(
            target, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"}
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(self.max_bytes + 1)
                return HttpResponse(
                    url=target,
                    final_url=response.geturl(),
                    status=response.status,
                    content_type=response.headers.get("Content-Type", ""),
                    body=body[: self.max_bytes],
                )
        except urllib.error.HTTPError as exc:
            transient = exc.code >= 500 or exc.code == 429
            raise ToolError(f"HTTP {exc.code} fetching {target}", retryable=transient) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ToolError(f"Could not fetch {target}: {reason}", retryable=True) from None
