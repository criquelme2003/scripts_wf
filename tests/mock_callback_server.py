"""Servidor HTTP mínimo para pruebas: registra cada POST recibido en un archivo
JSON-lines, para que un test externo pueda verificar qué llegó y cuándo."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
LOG_PATH = sys.argv[2] if len(sys.argv) > 2 else "mock_callback_received.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        record = {
            "received_at": time.time(),
            "authorization": self.headers.get("Authorization", ""),
            "body": json.loads(body.decode()) if body else None,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
