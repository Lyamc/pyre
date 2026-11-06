#!/usr/bin/env python3
"""OpenAI-compatible proxy and one-shot client for pyre's resume API.

pi does not need a patch: point its provider at `proxy`, which talks to pyre,
applies connect/idle/total timeouts, and continues a stream that died without a
successful finish_reason. See docs/streaming-resume.md.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

SUCCESS_FINISH = {"stop", "length", "tool_calls", "content_filter", "function_call"}


class Acc:
    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.token: str | None = None
        self.success = False
        self.last_error: dict | None = None

    def note_id(self, obj: dict) -> None:
        rid = obj.get("id") or (obj.get("error") or {}).get("resume_token") or (obj.get("pyre") or {}).get("resume_token")
        if isinstance(rid, str) and rid:
            self.token = rid


def kind_of(payload: str) -> str:
    """Classify one SSE data payload.

    success  — finish_reason the client may keep
    fail     — finish_reason error / abort, or any other non-success reason
    error    — OpenAI error object (no choices)
    done     — the [DONE] sentinel
    delta    — a content / reasoning / tool delta
    other    — comments are not passed here; usage-only chunks land here
    """
    if payload == "[DONE]":
        return "done"
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return "other"
    if not isinstance(obj, dict):
        return "other"
    if "error" in obj and "choices" not in obj:
        return "error"
    choices = obj.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "other"
    reason = choices[0].get("finish_reason")
    if isinstance(reason, str):
        return "success" if reason in SUCCESS_FINISH else "fail"
    return "delta"


def visible_delta(delta: dict, acc: Acc, replay: bool) -> dict:
    """Return the part of `delta` the downstream client has not already seen.

    Replay chunks from pyre repeat the whole prefix (or a suffix). Live chunks
    are already incremental.
    """
    out: dict = {}
    for key, attr in (("content", "content"), ("reasoning_content", "reasoning")):
        piece = delta.get(key)
        if not isinstance(piece, str) or piece == "":
            continue
        if replay:
            have = getattr(acc, attr)
            if piece.startswith(have):
                piece = piece[len(have):]
            elif have.startswith(piece) or piece in have:
                piece = ""
        if piece:
            setattr(acc, attr, getattr(acc, attr) + piece)
            out[key] = piece
    for key, value in delta.items():
        if key not in ("content", "reasoning_content"):
            out[key] = value
    return out


def _request(url: str, body: dict | None, headers: dict, method: str, connect: float) -> tuple[int, dict, object]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=connect)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw.decode() or "null")
        except json.JSONDecodeError:
            parsed = raw.decode(errors="replace")
        return e.code, dict(e.headers.items()), parsed
    return resp.status, dict(resp.headers.items()), resp


def iter_sse(resp, idle: float, deadline: float) -> Iterator[tuple[str, dict | None]]:
    """Yield ('data', obj) and ('comment', None). Raise TimeoutError on silence or total deadline."""
    sock = getattr(getattr(resp, "fp", None), "raw", None)
    underlying = getattr(sock, "_sock", sock)
    if underlying is not None and hasattr(underlying, "settimeout"):
        underlying.settimeout(idle)
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("total timeout")
        try:
            line = resp.readline()
        except (socket.timeout, TimeoutError) as e:
            raise TimeoutError("idle timeout") from e
        if line == b"":
            return
        if line in (b"\n", b"\r\n"):
            continue
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        if text.startswith(":"):
            yield "comment", None
            continue
        if text.startswith("data:"):
            payload = text[5:].lstrip()
            if payload == "[DONE]":
                yield "data", {"__done__": True}
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield "data", obj


def stream_once(url: str, body: dict, headers: dict, connect: float, idle: float, deadline: float) -> tuple[int, dict, list[dict]]:
    """POST and return (status, response headers, list of event dicts). Status 0 means the socket died."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={**headers, "Accept": "text/event-stream", "Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=connect)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            parsed = {"error": {"message": raw.decode(errors="replace"), "type": "server_error"}}
        return e.code, dict(e.headers.items()), [parsed] if isinstance(parsed, dict) else []
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
        return 0, {}, [{"error": {"message": str(e), "type": "server_error", "code": "connect_error"}}]
    events: list[dict] = []
    try:
        for kind, obj in iter_sse(resp, idle, deadline):
            if kind == "comment":
                events.append({"__comment__": True})
                continue
            assert obj is not None
            events.append(obj)
            if obj.get("__done__"):
                break
            payload = json.dumps(obj)
            if kind_of(payload) in ("success", "fail"):
                # Keep reading so a following [DONE] is consumed, but stop after a short grace.
                pass
    except TimeoutError as e:
        events.append({"error": {"message": str(e), "type": "server_error", "code": "timeout"}})
    except (OSError, urllib.error.URLError) as e:
        events.append({"error": {"message": str(e), "type": "server_error", "code": "connection_lost"}})
    finally:
        try:
            resp.close()
        except OSError:
            pass
    return 200, dict(resp.headers.items()), events


def _finish_ok(events: list[dict]) -> bool:
    saw = False
    for ev in events:
        if ev.get("__done__") or ev.get("__comment__"):
            continue
        k = kind_of(json.dumps(ev))
        if k == "success":
            saw = True
        elif k in ("fail", "error"):
            return False
    return saw


def complete_chat(base: str, body: dict, headers: dict, connect: float, idle: float, total: float, attempts: int) -> tuple[Acc, list[dict]]:
    """Run one chat, resuming aborted generations until it succeeds or attempts run out.

    Returns the accumulated text and the downstream events (already deduped).
    """
    url = base.rstrip("/") + "/v1/chat/completions"
    deadline = time.monotonic() + total
    acc = Acc()
    downstream: list[dict] = []
    status, hdrs, events = _with_connect_retries(url, body, headers, connect, idle, deadline, attempts)
    token = hdrs.get("x-resume-token") or hdrs.get("X-Resume-Token")
    _absorb(events, acc, downstream, replay=False)
    if token:
        acc.token = token
    if acc.success or not acc.token:
        return acc, downstream
    for _ in range(attempts):
        if time.monotonic() > deadline:
            break
        time.sleep(0.4)
        resume_body = {"resume": acc.token, "stream": True, "continue": True, "replay": True}
        status, _, events = stream_once(url, resume_body, headers, connect, idle, deadline)
        if status == 409:
            continue
        if status == 404:
            downstream.append({"error": {"message": "generation not found", "type": "server_error", "code": "generation_not_found", "resume_token": acc.token}})
            break
        before = acc.success
        _absorb(events, acc, downstream, replay=True)
        if acc.success or before:
            break
        # A hard error (not aborted) is terminal.
        if any(kind_of(json.dumps(ev)) == "fail" and (ev.get("pyre") or {}).get("status") == "error" for ev in events if isinstance(ev, dict)):
            break
    return acc, downstream


def _with_connect_retries(url, body, headers, connect, idle, deadline, attempts):
    delay = 0.5
    last = (0, {}, [{"error": {"message": "no attempt", "type": "server_error"}}])
    for i in range(max(1, attempts)):
        if time.monotonic() > deadline:
            break
        status, hdrs, events = stream_once(url, body, headers, connect, idle, deadline)
        last = (status, hdrs, events)
        if status != 0:
            return last
        # Only retry a connect failure before any token was produced.
        if any(isinstance(ev, dict) and (ev.get("choices") or ev.get("id")) for ev in events):
            return last
        time.sleep(delay)
        delay = min(delay * 2, 4.0)
    return last


def _absorb(events: list[dict], acc: Acc, downstream: list[dict], replay: bool) -> None:
    for ev in events:
        if not isinstance(ev, dict) or ev.get("__comment__"):
            continue
        if ev.get("__done__"):
            if acc.success:
                downstream.append(ev)
            continue
        acc.note_id(ev)
        payload = json.dumps(ev)
        k = kind_of(payload)
        if k == "delta":
            delta = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = visible_delta(delta, acc, replay=(replay and bool((ev.get("pyre") or {}).get("replay"))))
            if not piece:
                continue
            cloned = json.loads(payload)
            cloned["choices"][0]["delta"] = piece
            if replay:
                cloned.setdefault("pyre", {})["replay"] = False
            downstream.append(cloned)
        elif k == "success":
            acc.success = True
            downstream.append(ev)
        elif k in ("fail", "error"):
            # Held back. A later resume may still finish the generation; the
            # caller forwards last_error only if nothing succeeded.
            acc.last_error = ev
        else:
            downstream.append(ev)


def _fail_frame(message: str, token: str | None) -> dict:
    return {
        "error": {"message": message, "type": "server_error", "code": "generation_aborted", "resume_token": token},
        "pyre": {"status": "aborted", "resume_token": token, "resumable": bool(token)},
    }


class Proxy(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:8099"
    connect = 10.0
    idle = 20.0
    total = 1800.0
    attempts = 5

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _headers(self) -> dict:
        out = {}
        for key in ("authorization", "x-api-key", "content-type"):
            value = self.headers.get(key)
            if value:
                out[key] = value
        return out

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET", None)

    def do_POST(self) -> None:  # noqa: N802
        raw = self._read_body()
        path = self.path.split("?", 1)[0]
        if path.rstrip("/") in ("/v1/chat/completions", "/chat/completions"):
            self._chat(raw)
            return
        self._forward("POST", raw)

    def _forward(self, method: str, raw: bytes | None) -> None:
        url = self.upstream.rstrip("/") + self.path
        headers = self._headers()
        data = raw if raw else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=self.connect)
            body = resp.read()
            status = resp.status
            rh = dict(resp.headers.items())
        except urllib.error.HTTPError as e:
            body = e.read()
            status = e.code
            rh = dict(e.headers.items())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            msg = json.dumps({"error": {"message": str(e), "type": "server_error", "code": "connect_error"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(status)
        ctype = rh.get("Content-Type") or rh.get("content-type") or "application/json"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chat(self, raw: bytes) -> None:
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        if not isinstance(body, dict):
            body = {}
        headers = self._headers()
        base = self.upstream
        # Preserve the caller's stream flag. Non-stream is one JSON object.
        if not body.get("stream"):
            self._forward("POST", raw)
            return
        acc, events = complete_chat(base, body, headers, self.connect, self.idle, self.total, self.attempts)
        if acc.success and not any(isinstance(ev, dict) and ev.get("__done__") for ev in events):
            events.append({"__done__": True})
        if not acc.success:
            events.append(acc.last_error or _fail_frame("generation interrupted and resume did not finish", acc.token))
            events.append({"__done__": True})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        if acc.token:
            self.send_header("X-Request-Id", acc.token)
            self.send_header("X-Resume-Token", acc.token)
        self.end_headers()
        for ev in events:
            if ev.get("__comment__"):
                self.wfile.write(b": pyre-ping\n\n")
            elif ev.get("__done__"):
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(b"data: " + json.dumps(ev, separators=(",", ":")).encode() + b"\n\n")
            self.wfile.flush()


def cmd_proxy(args: argparse.Namespace) -> int:
    Proxy.upstream = args.upstream
    Proxy.connect = args.connect_timeout
    Proxy.idle = args.idle_timeout
    Proxy.total = args.total_timeout
    Proxy.attempts = args.resume_attempts
    host, _, port = args.listen.rpartition(":")
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Proxy)
    print(f"pyre resume proxy on http://{host or '127.0.0.1'}:{port} -> {args.upstream}", flush=True)
    print(f"timeouts: connect {args.connect_timeout:.0f}s, idle {args.idle_timeout:.0f}s, total {args.total_timeout:.0f}s", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "stream": True,
        "max_tokens": args.max_tokens,
    }
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    acc, events = complete_chat(args.upstream, body, headers, args.connect_timeout, args.idle_timeout, args.total_timeout, args.resume_attempts)
    if acc.reasoning:
        sys.stderr.write(acc.reasoning + "\n")
    sys.stdout.write(acc.content)
    if acc.content and not acc.content.endswith("\n"):
        sys.stdout.write("\n")
    if not acc.success:
        err = next((ev.get("error", {}).get("message") for ev in reversed(events) if isinstance(ev, dict) and ev.get("error")), "stream ended without a successful finish_reason")
        sys.stderr.write(f"pyre: {err}\n")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="pyre chat resume proxy / client")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--idle-timeout", type=float, default=20.0)
    p.add_argument("--total-timeout", type=float, default=1800.0)
    p.add_argument("--resume-attempts", type=int, default=5)
    sub = p.add_subparsers(dest="cmd", required=True)
    proxy = sub.add_parser("proxy", help="listen locally and forward to pyre, resuming dropped streams")
    proxy.add_argument("--upstream", default="http://192.168.0.30:8099")
    proxy.add_argument("--listen", default="127.0.0.1:8100")
    proxy.set_defaults(func=cmd_proxy)
    one = sub.add_parser("complete", help="one streaming chat; exit 1 if it cannot be finished")
    one.add_argument("--upstream", default="http://127.0.0.1:8099")
    one.add_argument("--model", default="")
    one.add_argument("--message", required=True)
    one.add_argument("--max-tokens", type=int, default=256)
    one.add_argument("--api-key", default="")
    one.set_defaults(func=cmd_complete)
    args = p.parse_args()
    return args.func(args)


def _self_test() -> None:
    assert kind_of("[DONE]") == "done"
    assert kind_of(json.dumps({"choices": [{"finish_reason": "stop"}]})) == "success"
    assert kind_of(json.dumps({"choices": [{"finish_reason": "error"}], "pyre": {"status": "aborted"}})) == "fail"
    assert kind_of(json.dumps({"error": {"message": "x"}})) == "error"
    assert kind_of(json.dumps({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]})) == "delta"
    acc = Acc()
    live = visible_delta({"content": "hel"}, acc, replay=False)
    assert live == {"content": "hel"} and acc.content == "hel"
    replay = visible_delta({"content": "hello"}, acc, replay=True)
    assert replay == {"content": "lo"} and acc.content == "hello"
    again = visible_delta({"content": "hello"}, acc, replay=True)
    assert "content" not in again
    # A stream of deltas then an error is not success; a stop is.
    events = [
        {"id": "chatcmpl-0123456789abcdef", "choices": [{"delta": {"content": "a"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "error"}], "pyre": {"status": "aborted"}},
    ]
    assert not _finish_ok(events)
    events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    # fail already returned... rebuild
    assert _finish_ok([{"choices": [{"finish_reason": "stop"}]}])
    assert not _finish_ok([{"choices": [{"finish_reason": "stop"}]}, {"error": {"message": "no"}}])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
        print("ok")
        raise SystemExit(0)
    raise SystemExit(main())
