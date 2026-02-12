"""Backward-compatible shim. Import from src.llm.client instead."""
from src.llm.client import LLMClient

__all__ = ["LLMClient"]
