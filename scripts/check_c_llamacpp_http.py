"""Smoke-test the C `llamacpp-complete` HTTP transport against a mock server.

Starts a tiny OpenAI-compatible mock on an ephemeral port, POSTs a request via
the C binary, and asserts two things:

1. Happy path: it extracts choices[0].message.content from a well-formed
   response.
2. Robustness: a malicious/buggy server returning adversarial bodies (non-JSON,
   missing/empty fields, an oversized 5 MB content, deeply-nested junk,
   truncated JSON) is handled by *failing closed* (clean non-zero exit, a
   diagnostic on stderr) — never a crash. Run under the ASan binary in CI this
   guards the untrusted-response path (see the JSON-parser recursion-depth fix).

Skips cleanly when the binary was built without libcurl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CANNED_CONTENT = (
    "<think>mock reasoning</think>"
    '{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"T-1"},"id":1}'
)

# Adversarial response bodies a buggy/malicious server might return; each must
# make llamacpp-complete fail closed (no crash).
_ADVERSARIAL = {
    "not_json": "<html>502 Bad Gateway</html>",
    "missing_choices": '{"id":"x"}',
    "choices_empty": '{"choices":[]}',
    "choice_no_content": '{"choices":[{"message":{}}]}',
    "content_huge": '{"choices":[{"message":{"content":"' + "A" * 5_000_000 + '"}}]}',
    "content_deep": '{"choices":[{"message":{"content":"x"}}],"j":' + "[" * 4000 + "]" * 4000 + "}",
    "truncated": '{"choices":[{"message":{"content":"x"',
}

_SANITIZER_MARKERS = (
    "AddressSanitizer",
    "stack-overflow",
    "runtime error",
    "heap-buffer-overflow",
    "use-after",
    "detected memory leaks",
)


def _make_server(body: str) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)  # drain the request body
            if self.path != "/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:  # silence access logging
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), _Handler)


def _run(toyforge_c: Path, body: str) -> subprocess.CompletedProcess[str]:
    server = _make_server(body)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "request.json"
            req.write_text(json.dumps({"model": "mock", "messages": [], "max_tokens": 16}))
            return subprocess.run(
                [
                    str(toyforge_c), "llamacpp-complete",
                    "--request", str(req),
                    "--base-url", f"http://127.0.0.1:{port}",
                    "--api-key", "no-key",
                ],
                check=False, text=True, capture_output=True,
            )  # fmt: skip
    finally:
        server.shutdown()
        thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    well_formed = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": CANNED_CONTENT}}]}
    )
    proc = _run(args.toyforge_c, well_formed)
    if proc.returncode == 3:
        print("SKIP: toyforge-c built without libcurl")
        return 0
    if proc.returncode != 0:
        print(f"llamacpp-complete failed (rc={proc.returncode}): {proc.stderr.strip()}")
        return 1
    if proc.stdout.strip() != CANNED_CONTENT:
        print(f"content mismatch:\n  got: {proc.stdout.strip()!r}\n  want: {CANNED_CONTENT!r}")
        return 1

    for label, body in _ADVERSARIAL.items():
        p = _run(args.toyforge_c, body)
        out = p.stdout + p.stderr
        if any(marker in out for marker in _SANITIZER_MARKERS):
            print(f"adversarial[{label}]: SANITIZER finding (rc={p.returncode})")
            for line in out.splitlines():
                if "ERROR" in line or "SUMMARY" in line:
                    print("   " + line)
            return 1
        # Fail-closed = a clean non-zero exit (not success, not a signal/crash).
        if not (1 <= p.returncode < 128):
            print(f"adversarial[{label}]: expected clean fail-closed, got rc={p.returncode}\n{out}")
            return 1

    print(
        "llamacpp-complete OK against mock: round-trip + "
        f"{len(_ADVERSARIAL)} adversarial responses fail closed (no crash)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
