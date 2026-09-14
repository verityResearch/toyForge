"""Smoke-test the C `llamacpp-eval` live two-stage decode loop against a mock.

The mock returns free-form reasoning for stage-1 (no-grammar) requests. For the
stage-2 (grammar-constrained) call it distinguishes the greedy pass
(temperature 0) from sampled passes (temperature > 0).

Two scenarios over the 1-trajectory fixture with --k 3, asserting the C
aggregation matches eval/score.aggregate:

  - unanimous: greedy fails; all 3 samples emit the SAME valid call ->
    pass@1=0, pass@k=1, pass@maj=1.
  - split vote: greedy fails; samples emit [BAD, BAD, GOOD] (two failing,
    one passing) -> pass@1=0, pass@k=1 (a sample passed) but pass@maj=0
    (the MAJORITY call class is the failing one). This distinguishes pass@maj
    from pass@k — the unanimous case alone cannot catch a pass@maj==pass@k bug.

Skips when built without libcurl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURE = "c/tests/fixtures/verify-data-one-split/train.jsonl"
THINK_CONTENT = "The object is in NEW; ticket_open.accepted moves it to TRIAGED."
GOOD_CALL = json.dumps(
    {
        "jsonrpc": "2.0",
        "method": "ticket_open",
        "params": {
            "requester_id": "user:c",
            "body_text": "Yw==",
            "lifecycle_profile": "std",
        },
        "id": 1,
    }
)
BAD_CALL = json.dumps(
    {
        "jsonrpc": "2.0",
        "method": "ticket_open",
        "params": {},
        "id": 1,
    }  # missing required -> schema fail
)


def _make_server(split: bool) -> ThreadingHTTPServer:
    # Sampled-call sequence: unanimous -> all GOOD; split -> [BAD, BAD, GOOD].
    state = {"sampled": 0}
    sampled_seq = [BAD_CALL, BAD_CALL, GOOD_CALL] if split else [GOOD_CALL, GOOD_CALL, GOOD_CALL]

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if "grammar" not in data:
                content = THINK_CONTENT  # stage-1 thinking
            elif float(data.get("temperature", 0.0)) == 0.0:
                content = BAD_CALL  # greedy decode -> fails
            else:
                i = state["sampled"]
                state["sampled"] += 1
                content = sampled_seq[i] if i < len(sampled_seq) else GOOD_CALL
            payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), _Handler)


def run_scenario(toyforge_c: Path, split: bool, expected: dict) -> tuple[bool, str]:
    """Returns (skip, error). skip=True when libcurl is absent."""
    server = _make_server(split)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "eval"
            proc = subprocess.run(
                [
                    str(toyforge_c), "llamacpp-eval",
                    "--data-path", FIXTURE,
                    "--schemas-dir", "schemas",
                    "--grammar-path", "schemas/jsonrpc.gbnf",
                    "--base-url", f"http://127.0.0.1:{port}",
                    "--k", "3",
                    "--out-prefix", str(prefix),
                    "--run-name", "mock-eval",
                    "--llamacpp-commit", "abc123",
                ],
                check=False, text=True, capture_output=True,
            )  # fmt: skip
            if proc.returncode == 3:
                return True, ""
            scorecard = json.loads(Path(f"{prefix}.json").read_text())
    finally:
        server.shutdown()
        thread.join(timeout=5)

    got = {key: scorecard.get(key) for key in expected}
    if got != expected:
        return False, f"split={split}: got={got} want={expected}\n{proc.stdout}{proc.stderr}"
    return False, ""


def run_k_guard_scenario(toyforge_c: Path) -> tuple[bool, str]:
    """A `--k` so large that k * sizeof(sample buffer) overflows size_t must fail
    closed (exit 2), not wrap to a small allocation and write past it. 2**51+1 is
    above SIZE_MAX/8192. Returns (skip, error)."""
    server = _make_server(split=False)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    str(toyforge_c), "llamacpp-eval",
                    "--data-path", FIXTURE,
                    "--schemas-dir", "schemas",
                    "--grammar-path", "schemas/jsonrpc.gbnf",
                    "--base-url", f"http://127.0.0.1:{port}",
                    "--k", str(2**51 + 1),
                    "--out-prefix", str(Path(tmp) / "eval"),
                    "--run-name", "k-guard",
                ],
                check=False, text=True, capture_output=True,
            )  # fmt: skip
    finally:
        server.shutdown()
        thread.join(timeout=5)
    if proc.returncode == 3:
        return True, ""
    out = proc.stdout + proc.stderr
    if "AddressSanitizer" in out or not (0 < proc.returncode < 128):
        return False, f"huge --k not failed-closed: rc={proc.returncode}\n{out}"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    scenarios = [
        (
            False,
            {
                "pass_at_1": 0.0,
                "pass_at_k": 1.0,
                "pass_at_maj": 1.0,
                "k": 3,
                "runtime": "llamacpp",
                "llamacpp_commit": "abc123",
            },
        ),
        (
            True,
            {
                "pass_at_1": 0.0,
                "pass_at_k": 1.0,
                "pass_at_maj": 0.0,
                "k": 3,
                "runtime": "llamacpp",
                "llamacpp_commit": "abc123",
            },
        ),
    ]
    for split, expected in scenarios:
        skip, err = run_scenario(args.toyforge_c, split, expected)
        if skip:
            print("SKIP: toyforge-c built without libcurl")
            return 0
        if err:
            print(err)
            return 1

    skip, err = run_k_guard_scenario(args.toyforge_c)
    if skip:
        print("SKIP: toyforge-c built without libcurl")
        return 0
    if err:
        print(err)
        return 1

    print(
        "llamacpp-eval k>1 aggregation OK against mock: unanimous (pass@maj=1) and "
        "split-vote (pass@k=1 but pass@maj=0 — majority is the failing call); "
        "overflow-sized --k fails closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
