"""Centralized Rich Console proxy + verbosity controls.

All toyForge modules import `console` from here rather than instantiating their
own. Because Python's `from X import Y` binds Y to the object that existed at
import time, we use a tiny proxy class that holds a swappable inner Console.
This lets `set_quiet(True)` change the underlying Console for every module
without requiring re-imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console


class _ConsoleProxy:
    """Thin wrapper that forwards all attribute access to a swappable inner Console.

    Forwarding includes `print`, `log`, `status`, etc. — anything Rich's Console
    exposes. The inner instance can be replaced via `_swap_inner()` without
    breaking any caller's reference to the proxy.
    """

    def __init__(self, inner: Console) -> None:
        # Use object.__setattr__ to avoid recursion through __setattr__ if any.
        object.__setattr__(self, "_inner", inner)

    def _swap_inner(self, new_inner: Console) -> None:
        object.__setattr__(self, "_inner", new_inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    # Common-path explicit delegations help static analysis and avoid surprises
    # with magic methods that bypass __getattr__:
    @property
    def quiet(self) -> bool:
        return object.__getattribute__(self, "_inner").quiet

    # Context manager protocol — Rich's Live/Progress use `with console:` internally.
    def __enter__(self) -> Any:
        return object.__getattribute__(self, "_inner").__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return object.__getattribute__(self, "_inner").__exit__(exc_type, exc_val, exc_tb)


console: _ConsoleProxy = _ConsoleProxy(Console())

_verbose: bool = False


def set_quiet(quiet: bool) -> None:
    """Replace the inner Console with one matching the quiet flag."""
    console._swap_inner(Console(quiet=quiet))


def set_verbose(verbose: bool) -> None:
    global _verbose
    _verbose = verbose


def is_verbose() -> bool:
    return _verbose


def warn_overwrite(paths: list[Path]) -> None:
    """Print a yellow warning if any of *paths* already exist on disk."""
    existing = [p for p in paths if p.exists()]
    if existing:
        console.print(
            f"[yellow]Overwriting existing files: {', '.join(p.name for p in existing)}[/yellow]"
        )
