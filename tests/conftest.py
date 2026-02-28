"""Shared pytest configuration: stub out heavy dependencies before imports."""

import sys
import types

# Stub out openai so LLMClient can be imported without the real package
_openai_stub = types.ModuleType("openai")
_openai_stub.OpenAI = object  # type: ignore[attr-defined]
sys.modules.setdefault("openai", _openai_stub)

# Stub out PyQt6 so GUI modules can be imported without the real package
for _mod in ("PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
