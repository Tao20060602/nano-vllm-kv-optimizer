"""Text-only OpenAI-compatible HTTP endpoint backed by the real NanoKV engine."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import EngineAdapter, Service, ServiceError


def make_server(service: Service, model: str, host="127.0.0.1", port=8080, api_key=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass  # Do not log prompts or Authorization headers.

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self):
            if api_key is None or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + api_key):
                return True
            self.send_json(401, {"error": {"message": "invalid API key"}})
            return False

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path == "/healthz":
                state = service.snapshot()
                self.send_json(200 if state["healthy"] else 503, state)
            elif self.path == "/metrics":
                body = service.metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json(404, {"error": {"message": "unknown endpoint"}})

        def sse(self, payload):
            self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n\n").encode())
            self.wfile.flush()

        def do_POST(self):
            if not self.authenticated():
                self.close_connection = True
                return
            if self.path != "/v1/chat/completions":
                self.close_connection = True
                self.send_json(404, {"error": {"message": "unknown endpoint"}})
                return
            job = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ServiceError("request body must be between 1 byte and 4 MiB", 413)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ServiceError("request must be a JSON object", 400)
                unknown = set(payload) - {"model", "messages", "max_tokens", "temperature", "stream", "stream_options", "timeout_s"}
                if unknown:
                    raise ServiceError(f"unsupported request fields: {sorted(unknown)}", 400)
                if payload.get("model", model) != model:
                    raise ServiceError("unknown model", 404)
                streaming = payload.get("stream", False)
                if not isinstance(streaming, bool):
                    raise ServiceError("stream must be boolean", 400)
                stream_options = payload.get("stream_options", {})
                if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
                    raise ServiceError("unsupported stream_options", 400)
                if "include_usage" in stream_options and not isinstance(stream_options["include_usage"], bool):
                    raise ServiceError("include_usage must be boolean", 400)
                job = service.submit(payload.get("messages"), payload.get("max_tokens", 64),
                                     payload.get("temperature", 0.0), payload.get("timeout_s", 300.0))
            except (ValueError, TypeError, ServiceError) as exc:
                self.close_connection = True
                self.send_json(getattr(exc, "status", 400), {"error": {"message": str(exc)}})
                return
            base = {"id": job.id, "model": model, "created": int(time.time())}
            try:
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    self.sse({**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                fragments = []
                while True:
                    try:
                        event = job.events.get(timeout=1)
                    except queue.Empty:
                        if streaming:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        continue
                    if event["type"] == "token":
                        fragments.append(event["text"])
                        if streaming:
                            self.sse({**base, "object": "chat.completion.chunk", "choices": [
                                {"index": 0, "delta": {"content": event["text"], "token_ids": event["token_ids"]},
                                 "finish_reason": None}]})
                        continue
                    if event["type"] == "error":
                        error = {"error": {"message": event["message"], "type": event["status"]}}
                        if streaming:
                            self.sse(error)
                        else:
                            self.send_json(400 if event["status"] == "invalid" else 503, error)
                    else:
                        usage = {"prompt_tokens": event["prompt_tokens"], "completion_tokens": event["output_tokens"],
                                 "total_tokens": event["prompt_tokens"] + event["output_tokens"]}
                        if streaming:
                            self.sse({**base, "object": "chat.completion.chunk", "choices": [
                                {"index": 0, "delta": {}, "finish_reason": event["finish_reason"]}]})
                            if stream_options.get("include_usage", False):
                                self.sse({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
                        else:
                            self.send_json(200, {**base, "object": "chat.completion", "usage": usage,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(fragments)},
                                             "finish_reason": event["finish_reason"]}]})
                    if streaming:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    break
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                job.cancelled.set()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve the real NanoKV engine with bounded FIFO admission")
    p.add_argument("--model", required=True, help="local model directory")
    p.add_argument("--engine-config", required=True, help="JSON object containing existing NanoKV Config options")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-outstanding", type=int, default=4, help="active + queued requests; execution remains serial")
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=131072)
    p.add_argument("--queue-timeout", type=float, default=60)
    args = p.parse_args(argv)
    options = json.loads(Path(args.engine_config).read_text())
    if not isinstance(options, dict):
        p.error("engine config must be a JSON object")
    service = Service(lambda: EngineAdapter(args.model, options), args.max_outstanding,
                      args.max_output_tokens, args.max_prompt_tokens, args.queue_timeout)
    print("Loading NanoKV in its engine owner thread...", flush=True)
    service.start()
    server = make_server(service, Path(args.model).name, args.host, args.port, os.getenv("NANOKV_API_KEY"))
    print(f"NanoKV listening at http://{args.host}:{args.port}; single_sequence_fifo", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
