"""Serve the Iris classifier over a small standard-library HTTP API."""

import argparse
import json
import os
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib

from src.paths import DEFAULT_MODEL_PATH
from src.predict import predict_species

MAX_REQUEST_BYTES = 16 * 1024


class PredictionHandler(BaseHTTPRequestHandler):
    model: Any

    def __init__(self, request, client_address, server, model: Any) -> None:
        self.model = model
        super().__init__(request, client_address, server)

    def _send_json(self, status: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/healthz", "/readyz"}:
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(400, {"error": "Request body must be between 1 and 16384 bytes"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object of measurements")
            species = predict_species(self.model, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(200, {"species": species})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    model = joblib.load(args.model_path)
    server = ThreadingHTTPServer((args.host, args.port), partial(PredictionHandler, model=model))
    print(f"Serving predictions on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
