"""Shared HTTP transport for provider adapters (stdlib only - no provider SDKs).

* Real request-level timeouts: one total deadline per request, enforced on the socket
  while connecting, waiting for headers and reading the body. The call runs in the caller's
  thread - no helper threads - so a timeout simply raises and nothing is left running.
* Only the headers the adapter passes are sent (no environment is forwarded); proxies from
  the standard *_PROXY variables are honoured by urllib.
* Errors are raised as transport exceptions; adapters map them to structured LLM errors.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_RESPONSE_BYTES = 20_000_000
_CHUNK = 65_536


class TransportTimeout(Exception):
    pass


class TransportNetworkError(Exception):
    pass


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class Transport(Protocol):
    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResult: ...


def _socket_of(response: Any) -> socket.socket | None:
    raw = getattr(getattr(response, "fp", None), "raw", None)
    return getattr(raw, "_sock", None)


class HttpTransport:
    """POST JSON with a hard total deadline."""

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResult:
        deadline = time.monotonic() + timeout
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        )

        def remaining() -> float:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TransportTimeout(f"no complete response within {timeout:g}s")
            return left

        try:
            try:
                response = urllib.request.urlopen(request, timeout=remaining())  # noqa: S310
            except urllib.error.HTTPError as exc:  # non-2xx: still a complete HTTP response
                body = exc.read(MAX_RESPONSE_BYTES) if exc.fp else b""
                return HttpResult(exc.code, body, {k.lower(): v for k, v in exc.headers.items()})
            with response:
                sock = _socket_of(response)
                chunks: list[bytes] = []
                size = 0
                while True:
                    if response.isclosed():  # http.client closes once the body is complete
                        break
                    left = remaining()
                    if sock is not None:
                        try:
                            sock.settimeout(left)
                        except OSError:  # already closed by http.client
                            break
                    # read1: return what has arrived so the deadline is checked between reads
                    # (read(n) would block until n bytes, defeating the deadline on slow drips)
                    reader = getattr(response, "read1", None) or response.read
                    chunk = reader(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise TransportNetworkError("response too large")
                    chunks.append(chunk)
                return HttpResult(response.status, b"".join(chunks),
                                  {k.lower(): v for k, v in response.headers.items()})
        except TransportTimeout:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise TransportTimeout(f"no complete response within {timeout:g}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TransportTimeout(f"no complete response within {timeout:g}s") from exc
            raise TransportNetworkError(str(exc.reason)) from exc
        except (ConnectionError, OSError) as exc:
            raise TransportNetworkError(f"{type(exc).__name__}: {exc}") from exc
