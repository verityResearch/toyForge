"""End-to-end smoke-test of the C `teacher-expand` loop against a mock server.

Runs the loop for BOTH provider shapes against a path-aware mock:
  - OpenAI-compatible (`/chat/completions`): {"choices":[{"message":{"content":...}}]}
  - Anthropic Messages (`/v1/messages`):      {"content":[{"type":"text","text":...}]}
The mock returns invalid JSON for the first request of each run (exercising the
json_decode rejection + retry) and a canned valid trajectory afterwards, so every
seed is eventually accepted. Each run proves: retry recovery (accepted == N,
rejected == 0), the _run_start sentinel + exactly one json_decode in
rejections.jsonl, trajectory_id dedup (the mock echoes one id), source/provenance
on accepted rows, test.jsonl = the hand seeds, and a verify-data round-trip.
Skips when toyforge-c was built without libcurl.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from toyforge.scenario_gen.seeds import load_seeds

SEEDS_PATH = Path("scenarios/seeds.yaml")
_VALID_TRAJECTORY = json.dumps(load_seeds(SEEDS_PATH)[0])


def _split_blocks(text: str, n: int) -> list[dict]:
    """Split `text` across n Anthropic text content-blocks (concatenation ==
    text). Exercises extract_anthropic_content's multi-block accumulation."""
    step = max(1, len(text) // n)
    chunks = [text[i : i + step] for i in range(0, len(text), step)] or [""]
    return [{"type": "text", "text": c} for c in chunks]


def _make_server(
    *, anthropic_blocks: int = 1, inject_bad_first: bool = True
) -> ThreadingHTTPServer:
    counter = {"n": 0}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            n = counter["n"]
            counter["n"] += 1
            body = "{ not valid json" if (inject_bad_first and n == 0) else _VALID_TRAJECTORY
            if self.path.endswith("/v1/messages"):  # Anthropic shape
                # When the body is the valid trajectory, optionally fan it out
                # across multiple text blocks so the C parser must reassemble it.
                if anthropic_blocks > 1 and body == _VALID_TRAJECTORY:
                    content = _split_blocks(body, anthropic_blocks)
                else:
                    content = [{"type": "text", "text": body}]
                envelope = {"content": content}
            else:  # OpenAI-compatible shape
                envelope = {"choices": [{"message": {"content": body}}]}
            payload = json.dumps(envelope).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), _Handler)


def run_case(toyforge_c: Path, provider: str, n_seeds: int) -> tuple[bool, list[str]]:
    """Run teacher-expand for one provider against a fresh mock; returns (skip, failures)."""
    server = _make_server()
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    failures: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cmd = [
                str(toyforge_c), "teacher-expand",
                "--seeds-path", str(SEEDS_PATH),
                "--schemas-dir", "schemas",
                "--out-dir", str(out),
                "--base-url", f"http://127.0.0.1:{port}",
                "--provider", provider,
                "--expansions-per-seed", "1",
                "--max-retries", "3",
                "--train-frac", "0.8",
                "--dev-frac", "0.1",
                "--shuffle-seed", "42",
            ]  # fmt: skip
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            if proc.returncode == 3:
                return True, []
            if proc.returncode != 0:
                return False, [f"[{provider}] rc={proc.returncode}: {proc.stderr or proc.stdout}"]

            m = re.search(
                r"accepted=(\d+) rejected=(\d+) train=(\d+) dev=(\d+) test=(\d+)", proc.stdout
            )
            if not m:
                return False, [f"[{provider}] could not parse summary: {proc.stdout!r}"]
            accepted, rejected, train_n, dev_n, test_n = (int(x) for x in m.groups())
            if accepted != n_seeds:
                failures.append(f"[{provider}] accepted={accepted} expected {n_seeds}")
            if rejected != 0:
                failures.append(f"[{provider}] rejected={rejected} expected 0")
            if test_n != n_seeds:
                failures.append(f"[{provider}] test={test_n} expected {n_seeds}")
            if train_n != int(n_seeds * 0.8) or dev_n != int(n_seeds * 0.1):
                failures.append(f"[{provider}] split train={train_n} dev={dev_n}")

            rej_lines = [
                json.loads(line)
                for line in (out / "rejected" / "rejections.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if not rej_lines or not rej_lines[0].get("_run_start"):
                failures.append(f"[{provider}] rejections.jsonl missing _run_start sentinel")
            rejections = [r for r in rej_lines if not r.get("_run_start")]
            if len(rejections) != 1 or rejections[0].get("reason") != "json_decode":
                failures.append(f"[{provider}] expected 1 json_decode rejection, got {rejections}")

            rows: list[dict] = []
            for split in ("train", "dev"):
                rows += [
                    json.loads(line)
                    for line in (out / f"{split}.jsonl").read_text().splitlines()
                    if line.strip()
                ]
            ids = [r.get("trajectory_id") for r in rows]
            if len(ids) != len(set(ids)):
                failures.append(f"[{provider}] trajectory_ids not unique after dedup")
            for r in rows:
                if r.get("source") != "teacher_expansion":
                    failures.append(f"[{provider}] row source={r.get('source')!r}")
                if "seed_trajectory_id" not in r.get("provenance", {}):
                    failures.append(f"[{provider}] row missing provenance/seed_trajectory_id")

            test_rows = [
                json.loads(line)
                for line in (out / "test.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if any(r.get("source") != "hand_seed" for r in test_rows):
                failures.append(f"[{provider}] test.jsonl rows not all source=hand_seed")

            vd_cmd = [
                str(toyforge_c), "verify-data",
                "--data-dir", str(out),
                "--schemas-dir", "schemas",
            ]  # fmt: skip
            vd = subprocess.run(vd_cmd, check=False, text=True, capture_output=True)
            if vd.returncode != 0:
                failures.append(f"[{provider}] verify-data failed: {vd.stderr or vd.stdout}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return False, failures


def run_multiblock_anthropic_case(toyforge_c: Path, n_seeds: int) -> tuple[bool, list[str]]:
    """Anthropic responses split across 4 text blocks: proves extract_anthropic_content
    reassembles the multi-block content correctly (the single-block mock never did)."""
    server = _make_server(anthropic_blocks=4, inject_bad_first=False)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    failures: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            proc = subprocess.run(
                [
                    str(toyforge_c), "teacher-expand",
                    "--seeds-path", str(SEEDS_PATH),
                    "--schemas-dir", "schemas",
                    "--out-dir", str(out),
                    "--base-url", f"http://127.0.0.1:{port}",
                    "--provider", "anthropic",
                    "--expansions-per-seed", "1",
                    "--max-retries", "1",
                    "--train-frac", "0.8",
                    "--dev-frac", "0.1",
                    "--shuffle-seed", "42",
                ],
                check=False, text=True, capture_output=True,
            )  # fmt: skip
            if proc.returncode == 3:
                return True, []
            if proc.returncode != 0:
                return False, [f"[anthropic/multiblock] rc={proc.returncode}: {proc.stderr}"]
            m = re.search(r"accepted=(\d+) rejected=(\d+)", proc.stdout)
            if not m:
                return False, [f"[anthropic/multiblock] no summary: {proc.stdout!r}"]
            accepted, rejected = int(m.group(1)), int(m.group(2))
            # All blocks reassemble to the valid trajectory -> every seed accepted.
            if accepted != n_seeds:
                failures.append(f"[anthropic/multiblock] accepted={accepted} expected {n_seeds}")
            if rejected != 0:
                failures.append(f"[anthropic/multiblock] rejected={rejected} expected 0")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return False, failures


def run_negative_frac_case(toyforge_c: Path, n_seeds: int) -> tuple[bool, list[str]]:
    """Negative split fractions must yield empty (0) splits — not a wrapped SIZE_MAX
    count from an undefined negative-double->size_t cast (regression for that fix)."""
    server = _make_server(inject_bad_first=False)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    failures: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    str(toyforge_c), "teacher-expand",
                    "--seeds-path", str(SEEDS_PATH),
                    "--schemas-dir", "schemas",
                    "--out-dir", str(Path(tmp) / "out"),
                    "--base-url", f"http://127.0.0.1:{port}",
                    "--provider", "local",
                    "--expansions-per-seed", "1",
                    "--max-retries", "1",
                    "--train-frac", "-1.0",
                    "--dev-frac", "-0.5",
                    "--shuffle-seed", "42",
                ],
                check=False, text=True, capture_output=True,
            )  # fmt: skip
            if proc.returncode == 3:
                return True, []
            if proc.returncode != 0:
                return False, [f"[neg-frac] rc={proc.returncode}: {proc.stderr}"]
            m = re.search(r"train=(\d+) dev=(\d+)", proc.stdout)
            if not m:
                return False, [f"[neg-frac] no summary: {proc.stdout!r}"]
            train_n, dev_n = int(m.group(1)), int(m.group(2))
            # Empty splits expected; crucially neither may exceed the accepted count
            # (a wrapped SIZE_MAX would be ~1.8e19).
            if train_n > n_seeds or dev_n > n_seeds:
                failures.append(
                    f"[neg-frac] absurd split train={train_n} dev={dev_n} (n={n_seeds})"
                )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return False, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    n_seeds = len(load_seeds(SEEDS_PATH))
    all_failures: list[str] = []
    for provider in ("local", "anthropic"):
        skip, failures = run_case(args.toyforge_c, provider, n_seeds)
        if skip:
            print("SKIP: toyforge-c built without libcurl")
            return 0
        all_failures += failures

    skip, failures = run_multiblock_anthropic_case(args.toyforge_c, n_seeds)
    if skip:
        print("SKIP: toyforge-c built without libcurl")
        return 0
    all_failures += failures

    skip, failures = run_negative_frac_case(args.toyforge_c, n_seeds)
    if skip:
        print("SKIP: toyforge-c built without libcurl")
        return 0
    all_failures += failures

    if all_failures:
        print("\n".join(all_failures))
        return 1
    print(
        f"teacher-expand loop OK against mock for both providers (openai-compat + anthropic): "
        f"accepted={n_seeds}, 1 json_decode, deduped, verify-data clean; "
        f"anthropic multi-block content reassembled (accepted={n_seeds})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
