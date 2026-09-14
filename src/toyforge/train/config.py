"""Training configuration — loaded from train_config.yaml."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. `override` wins on conflicts. Lists are NOT merged
    element-wise — override replaces wholesale (typical YAML inheritance semantics)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_with_inheritance(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config, recursively resolving `extends:` directives.

    `extends:` is a path relative to the file's parent directory. Detects
    cycles by tracking visited absolute paths.
    """
    _seen = _seen if _seen is not None else set()
    abs_path = path.resolve()
    if abs_path in _seen:
        raise ValueError(f"circular extends: {abs_path}")
    _seen = _seen | {abs_path}
    doc = yaml.safe_load(abs_path.read_text()) or {}
    base_ref = doc.pop("extends", None)
    if base_ref is None:
        return doc
    base_path = (abs_path.parent / base_ref).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"extends target not found: {base_path} (referenced by {abs_path})")
    base_doc = _load_with_inheritance(base_path, _seen)
    return _deep_merge(base_doc, doc)


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    dropout: float = 0.05


@dataclass(frozen=True)
class GRPOConfig:
    num_generations: int = 8  # k rollouts per prompt
    rollout_temperature: float = 0.7
    beta: float = 0.04  # KL coefficient
    reward_scaling: float = 1.0
    rubric_preset: str = "shaped"


@dataclass(frozen=True)
class LlamaCppTrainConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "local"
    api_key_env: str | None = None
    grammar_path: Path = Path("schemas/jsonrpc.gbnf")
    constrained: bool = True
    reference_adapter_scale: float = 0.0
    commit: str = ""


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    method: str
    output_dir: Path
    train_path: Path
    dev_path: Path
    lora: LoRAConfig
    context_train: int = 8192
    optimizer: str = "paged_adamw_8bit"
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-5
    num_epochs: int = 3
    attn_impl: str = "sdpa"
    seed: int = 42
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    grpo: GRPOConfig | None = None
    adapter_init: Path | None = None  # Phase 1 SFT adapter to warm-start GRPO
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    runtime: str = "transformers"
    llamacpp: LlamaCppTrainConfig | None = None


def load_train_config(path: Path) -> TrainConfig:
    doc: dict[str, Any] = _load_with_inheritance(Path(path))
    required = ["base_model", "method"]
    for r in required:
        if r not in doc:
            raise KeyError(f"required key {r!r} missing in {path}")

    lora_doc = doc.get("lora") or {}
    lora = LoRAConfig(
        r=lora_doc.get("r", 32),
        alpha=lora_doc.get("alpha", 64),
        target_modules=list(lora_doc.get("target_modules", LoRAConfig().target_modules)),
        dropout=lora_doc.get("dropout", 0.05),
    )

    known_keys = {
        "base_model",
        "method",
        "output_dir",
        "data",
        "lora",
        "context",
        "optimizer",
        "batch_size",
        "grad_accum",
        "lr",
        "num_epochs",
        "attn_impl",
        "seed",
        "warmup_ratio",
        "lr_scheduler_type",
        "grpo",
        "adapter_init",
        "report_to",
        "runtime",
        "llamacpp",
        "extends",  # popped during inheritance resolution; defensive include
    }
    extra = set(doc.keys()) - known_keys
    if extra:
        warnings.warn(
            f"Unknown keys in train config (typo?): {sorted(extra)}",
            stacklevel=2,
        )

    data_doc = doc.get("data") or {}
    context_doc = doc.get("context") or {}
    llamacpp_doc = doc.get("llamacpp") or {}
    return TrainConfig(
        base_model=doc["base_model"],
        method=doc["method"],
        output_dir=Path(doc.get("output_dir", "out/run")),
        train_path=Path(data_doc.get("train", "data/train.jsonl")),
        dev_path=Path(data_doc.get("dev", "data/dev.jsonl")),
        lora=lora,
        context_train=context_doc.get("train", 8192),
        optimizer=doc.get("optimizer", "paged_adamw_8bit"),
        batch_size=doc.get("batch_size", 1),
        grad_accum=doc.get("grad_accum", 16),
        lr=doc.get("lr", 2e-5),
        num_epochs=doc.get("num_epochs", 3),
        attn_impl=doc.get("attn_impl", "sdpa"),
        seed=doc.get("seed", 42),
        warmup_ratio=doc.get("warmup_ratio", 0.05),
        lr_scheduler_type=doc.get("lr_scheduler_type", "cosine"),
        grpo=GRPOConfig(**(doc.get("grpo") or {})) if doc.get("grpo") is not None else None,
        adapter_init=Path(doc["adapter_init"]) if doc.get("adapter_init") else None,
        report_to=list(doc["report_to"]) if "report_to" in doc else ["tensorboard"],
        runtime=doc.get("runtime", "transformers"),
        llamacpp=(
            LlamaCppTrainConfig(
                base_url=llamacpp_doc.get("base_url", "http://localhost:8080/v1"),
                model=llamacpp_doc.get("model", doc["base_model"]),
                api_key_env=llamacpp_doc.get("api_key_env"),
                grammar_path=Path(llamacpp_doc.get("grammar_path", "schemas/jsonrpc.gbnf")),
                constrained=llamacpp_doc.get("constrained", True),
                reference_adapter_scale=llamacpp_doc.get("reference_adapter_scale", 0.0),
                commit=llamacpp_doc.get("commit", ""),
            )
            if doc.get("runtime", "transformers") == "llamacpp" or doc.get("llamacpp") is not None
            else None
        ),
    )
