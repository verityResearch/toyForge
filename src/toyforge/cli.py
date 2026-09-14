"""toyForge CLI — entry points invoked by the justfile."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from toyforge._console import console, set_quiet, set_verbose
from toyforge.scenario_gen.expand import expand_seeds
from toyforge.scenario_gen.seeds import load_seeds
from toyforge.scenario_gen.teachers.base import TeacherConfig, build_teacher
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

app = typer.Typer(add_completion=False, no_args_is_help=True)

_EXPAND_GRAMMAR_PATH_OPTION = typer.Option(
    None, "--grammar-path", help="GBNF grammar file for local llama.cpp structured output."
)
_EVAL_GRAMMAR_PATH_OPTION = typer.Option(
    Path("schemas/jsonrpc.gbnf"), "--grammar-path", help="GBNF grammar path for llama.cpp."
)
_MERGE_EXPORT_CONVERT_SCRIPT_OPTION = typer.Option(
    Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    "--convert-script",
    help="Path to llama.cpp's convert_hf_to_gguf.py.",
)


@app.callback()
def _global_options(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output."),
) -> None:
    """Global flags applied to every command."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")
    if quiet:
        set_quiet(True)
    if verbose:
        set_verbose(True)


@app.command()
def verify_seeds(
    seeds_path: Path = Path("scenarios/seeds.yaml"),
    schemas_dir: Path = Path("schemas"),
) -> None:
    """Verify every entry in scenarios/seeds.yaml passes the verifier."""
    schemas = load_schemas(schemas_dir)
    seeds = load_seeds(seeds_path)
    failed = 0
    for s in seeds:
        outputs = [
            f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"
            for step in s["steps"]
        ]
        r = verify_trajectory(s, outputs, schemas)
        if not r.passed:
            console.print(f"[red]FAIL[/red] {s['trajectory_id']}: {r.error_message}")
            failed += 1
    console.print(f"[bold]{len(seeds) - failed}/{len(seeds)} seeds pass[/bold]")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def verify_data(
    data_dir: Path = Path("data"),
    schemas_dir: Path = Path("schemas"),
) -> None:
    """Verify every trajectory in data/{train,dev,test}.jsonl passes the verifier.

    Skips files that don't exist (e.g., during partial-pipeline runs).
    """
    schemas = load_schemas(schemas_dir)
    overall_failed = 0
    for name in ["train", "dev", "test"]:
        p = data_dir / f"{name}.jsonl"
        if not p.exists():
            console.print(f"[dim]skip {p} (missing)[/dim]")
            continue
        ok, total = 0, 0
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError as e:
                console.print(f"[red]malformed line in {p.name}: {e}[/red]")
                overall_failed += 1
                total += 1
                continue
            outs = [
                f"<think>{s['thinking']}</think>{json.dumps(s['tool_call'])}" for s in traj["steps"]
            ]
            r = verify_trajectory(traj, outs, schemas)
            total += 1
            if r.passed:
                ok += 1
            else:
                overall_failed += 1
                tid = traj.get("trajectory_id", f"line-{total}")
                console.print(f"[red]FAIL[/red] {tid}: {r.error_message}")
        color = "green" if ok == total else "red"
        console.print(f"[{color}]{name}: {ok}/{total} pass[/{color}]")
    if overall_failed:
        raise typer.Exit(code=1)


@app.command(name="expand-from-seeds")
def expand_from_seeds(
    seeds_path: Path = Path("scenarios/smoke-tiny-seeds.yaml"),
    schemas_dir: Path = Path("schemas"),
    out_dir: Path = Path("data/smoke-tiny"),
) -> None:
    """Skip teacher expansion: write hand-seeds directly as train/dev/test.jsonl.

    Used for `just smoke-tiny` to avoid API calls. The same seeds appear in all
    three splits — this is a pipeline-validity check, NOT a real eval.
    """
    from toyforge.scenario_gen.provenance import hand_seed_provenance

    try:
        from toyforge import __version__ as _TOYFORGE_VERSION
    except ImportError:
        _TOYFORGE_VERSION = "0.0.0"

    load_schemas(schemas_dir)  # validates schemas/ structure; raises on bad schema
    seeds = load_seeds(seeds_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tag every seed row with hand-seed provenance, then write to all three splits.
    # (Deliberately repeating the same 5 rows across splits — pipeline sanity only.)
    for s in seeds:
        s["source"] = "hand_seed"
        s["provenance"] = hand_seed_provenance(
            seed_trajectory_id=s["trajectory_id"],
            toyforge_version=_TOYFORGE_VERSION,
        )

    for name in ["train", "dev", "test"]:
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for s in seeds:
                f.write(json.dumps(s) + "\n")

    console.print(f"[green]Wrote {len(seeds)} seeds → {out_dir}/{{train,dev,test}}.jsonl[/green]")


@app.command()
def expand(
    seeds_path: Path = Path("scenarios/seeds.yaml"),
    schemas_dir: Path = Path("schemas"),
    out_dir: Path = Path("data"),
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    expansions_per_seed: int = 25,
    shuffle_seed: int = 42,
    base_url: str | None = None,
    api_key_env: str | None = None,
    grammar_path: Path | None = _EXPAND_GRAMMAR_PATH_OPTION,
    json_response: bool = typer.Option(
        False,
        "--json-response",
        help="Request JSON-object response_format from OpenAI-compatible teachers.",
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from data/accepted_partial.jsonl if present."
    ),
    num_workers: int = typer.Option(
        10, "--num-workers", help="ThreadPoolExecutor worker count for parallel API calls."
    ),
) -> None:
    """Run teacher-driven expansion and write data/{train,dev,test}.jsonl."""
    cfg = TeacherConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        grammar_path=grammar_path,
        response_format={"type": "json_object"} if json_response else None,
    )
    teacher = build_teacher(cfg)
    stats = expand_seeds(
        seeds_path=seeds_path,
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=expansions_per_seed,
        out_dir=out_dir,
        shuffle_seed=shuffle_seed,
        resume=resume,
        num_workers=num_workers,
    )
    table = Table(title="Expansion stats")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)


# --- Training + eval commands (Phase 1+) ---


@app.command()
def train(
    config_path: Path = Path("train_configs/phase1-sft.yaml"),
) -> None:
    """Run training with the config at `config_path`. Dispatches by `method`."""
    from toyforge.train.config import load_train_config

    cfg = load_train_config(config_path)

    if cfg.method in {"vanilla", "sft_go"}:
        from toyforge.train.sft import run_sft

        if cfg.method == "sft_go":
            console.print(
                "[yellow]Warning:[/yellow] method='sft_go' currently routes to plain "
                "SFT — token-group weighting is not yet implemented (Phase 3 ablation)."
            )
        adapter_dir = run_sft(cfg)
        console.print(f"[green]Adapter saved:[/green] {adapter_dir}")
    elif cfg.method == "grpo":
        from toyforge.train.grpo import run_grpo

        try:
            adapter_dir = run_grpo(cfg)
        except (RuntimeError, NotImplementedError, FileNotFoundError, ValueError) as e:
            console.print(f"[red]GRPO unavailable:[/red] {e}")
            raise typer.Exit(code=1) from e
        console.print(f"[green]Adapter saved:[/green] {adapter_dir}")
    elif cfg.method in {"dpo", "orpo"}:
        console.print(
            f"[red]{cfg.method.upper()} trainer not yet implemented.[/red] "
            "Set method to 'vanilla' or 'sft_go'."
        )
        raise typer.Exit(code=1)
    else:
        console.print(f"[red]unknown training method: {cfg.method!r}[/red]")
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_cmd(
    adapter_dir: Path = Path("out/phase1-sft/adapter"),
    base_model: str = "Qwen/Qwen3-8B-Instruct",
    test_path: Path = Path("data/test.jsonl"),
    schemas_dir: Path = Path("schemas"),
    out_dir: Path = Path("reports"),
    run_name: str = "phase1-sft",
    k: int = 8,
    attn_impl: str = "sdpa",
    resume: bool = typer.Option(
        False, "--resume", help="Resume from reports/{run_name}_partial.jsonl if present."
    ),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens", help="Max tokens per step."),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Sampling temperature for k sampled outputs."
    ),
    runtime: str = typer.Option("transformers", "--runtime", help="transformers or llamacpp."),
    constrained: bool = typer.Option(
        False, "--constrained/--no-constrained", help="Use llama.cpp grammar-constrained JSON."
    ),
    llamacpp_base_url: str = typer.Option(
        "http://localhost:8080/v1", "--llamacpp-base-url", help="llama.cpp server /v1 URL."
    ),
    llamacpp_model: str | None = typer.Option(
        None, "--llamacpp-model", help="Model id served by llama.cpp."
    ),
    grammar_path: Path = _EVAL_GRAMMAR_PATH_OPTION,
    llamacpp_commit: str = typer.Option("", "--llamacpp-commit", help="llama.cpp git SHA."),
) -> None:
    """Evaluate a trained adapter and write reports/{run_name}.{json,md}."""
    from toyforge.eval import run_eval

    run_eval(
        adapter_dir=adapter_dir,
        test_path=test_path,
        schemas_dir=schemas_dir,
        out_dir=out_dir,
        base_model=base_model,
        run_name=run_name,
        k=k,
        attn_impl=attn_impl,
        resume=resume,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        runtime=runtime,
        constrained=constrained,
        llamacpp_base_url=llamacpp_base_url,
        llamacpp_model=llamacpp_model,
        grammar_path=grammar_path,
        llamacpp_commit=llamacpp_commit,
    )


@app.command(name="eval-baseline")
def eval_baseline(
    base_model: str = "Qwen/Qwen3-8B-Instruct",
    test_path: Path = Path("data/test.jsonl"),
    schemas_dir: Path = Path("schemas"),
    out_dir: Path = Path("reports"),
    run_name: str = "B0-zero-shot",
    k: int = 8,
    attn_impl: str = "sdpa",
    resume: bool = typer.Option(
        False, "--resume", help="Resume from reports/{run_name}_partial.jsonl if present."
    ),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens", help="Max tokens per step."),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Sampling temperature for k sampled outputs."
    ),
    runtime: str = typer.Option("transformers", "--runtime", help="transformers or llamacpp."),
    constrained: bool = typer.Option(
        False, "--constrained/--no-constrained", help="Use llama.cpp grammar-constrained JSON."
    ),
    llamacpp_base_url: str = typer.Option(
        "http://localhost:8080/v1", "--llamacpp-base-url", help="llama.cpp server /v1 URL."
    ),
    llamacpp_model: str | None = typer.Option(
        None, "--llamacpp-model", help="Model id served by llama.cpp."
    ),
    grammar_path: Path = _EVAL_GRAMMAR_PATH_OPTION,
    llamacpp_commit: str = typer.Option("", "--llamacpp-commit", help="llama.cpp git SHA."),
) -> None:
    """Evaluate the base model only (no adapter, no system prompt augmentation).
    Useful for B0 baseline in Phase 3 ablations."""
    from toyforge.eval import run_eval

    run_eval(
        adapter_dir=None,  # base-only
        test_path=test_path,
        schemas_dir=schemas_dir,
        out_dir=out_dir,
        base_model=base_model,
        run_name=run_name,
        k=k,
        attn_impl=attn_impl,
        resume=resume,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        runtime=runtime,
        constrained=constrained,
        llamacpp_base_url=llamacpp_base_url,
        llamacpp_model=llamacpp_model,
        grammar_path=grammar_path,
        llamacpp_commit=llamacpp_commit,
    )


@app.command()
def compare(
    reports_dir: Path = Path("reports"),
    out_path: Path = Path("reports/compare.md"),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Run name to use as Δpass@1 reference."
    ),
    last: int | None = typer.Option(
        None, "--last", help="Compare only the N most recently modified reports."
    ),
    pattern: str | None = typer.Option(
        None,
        "--pattern",
        help=(
            "Glob filter on report filenames, e.g. 'phase3-*.json'. "
            "Must include the .json extension. Defaults to *.json."
        ),
    ),
) -> None:
    """Roll up all reports/{name}.json into reports/compare.md."""
    from toyforge.eval.compare import compare_runs

    compare_runs(reports_dir, out_path, baseline=baseline, last=last, pattern=pattern)


@app.command(name="merge-export")
def merge_export_cmd(
    adapter_dir: Path = Path("out/phase1-sft/adapter"),
    base_model: str = "Qwen/Qwen3-8B-Instruct",
    out_dir: Path = Path("out/export"),
    outtype: str = typer.Option("f16", "--outtype", help="GGUF dtype: f16 or bf16."),
    quant: str = typer.Option(
        "none", "--quant", help="Post-convert quantize: 'none' (ship f16) or 'q8_0'."
    ),
    convert_script: Path = _MERGE_EXPORT_CONVERT_SCRIPT_OPTION,
    quantize_bin: str = typer.Option(
        "llama-quantize", "--quantize-bin", help="llama-quantize binary (for --quant q8_0)."
    ),
    dtype: str = typer.Option("bfloat16", "--dtype", help="HF load dtype for the merge."),
    attn_impl: str = typer.Option("sdpa", "--attn-impl", help="Attention impl for the load."),
    python_bin: str = typer.Option(
        "python3", "--python-bin", help="Python that runs the converter (needs the gguf dep)."
    ),
) -> None:
    """Merge a LoRA adapter into its base and export a GGUF for the strict-C runtime.

    The deploy seam between toyForge (PyTorch/PEFT) and the strict-C runtime (merged-GGUF only):
    PEFT merge_and_unload → convert_hf_to_gguf.py → (optional) llama-quantize.
    """
    import subprocess  # noqa: PLC0415

    from toyforge.export import merge_export

    try:
        merge_export(
            base_model=base_model,
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            outtype=outtype,
            quant=quant,
            convert_script=convert_script,
            quantize_bin=quantize_bin,
            dtype=dtype,
            attn_impl=attn_impl,
            python_bin=python_bin,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ImportError) as e:
        console.print(f"[red]merge-export failed:[/red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
