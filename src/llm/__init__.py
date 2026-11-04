"""LLM package exports.

Importing this package must stay cheap: the GUI imports the LLM client on the
startup path, and ``openai`` alone used to cost ~364ms (F-18).  Exports are
resolved lazily on first attribute access and the submodule import inside
``__getattr__`` is the only place that pulls the heavy module in.
"""

from typing import TYPE_CHECKING

__all__ = ["LLMClient", "ResponseTruncatedError"]

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .client import LLMClient, ResponseTruncatedError


def __getattr__(name: str):
    if name in ("LLMClient", "ResponseTruncatedError"):
        from . import client as _client
        value = getattr(_client, name)
        # Cache on the package module so later lookups skip __getattr__.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
