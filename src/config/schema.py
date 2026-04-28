"""Typed configuration schema for Skyrim XML Translator."""

from dataclasses import dataclass, field, fields, asdict
from typing import Any


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    max_retries: int = 3
    backoff_base: float = 0.5
    stream: bool = False
    request_timeout: int = 30
    request_timeout_step: int = 15
    request_timeout_max: int = 180
    parameters: dict = field(default_factory=lambda: {
        "temperature": None,
        "top_p": None,
        "enable_thinking": None,
        "reasoning_effort": None,
    })


@dataclass
class EmbeddingConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-large"
    dimensions: int = 1536


@dataclass
class RAGConfig:
    similarity_threshold: float = 0.75
    short_term_max_results: int = 5
    long_term_max_results: int = 2
    short_term_max_chars: int = 32
    keyword_max_queries: int = 128
    keyword_task_decompose_enabled: bool = True
    keyword_task_keep_original: bool = False
    min_vector_score: float = 0.45
    keyword_weight_enabled: bool = True
    keyword_weight_candidate_pool_size: int = 24
    keyword_weight_keep_k: int = 24
    keyword_weight_min_primary_hits: int = 8
    keyword_weight_exact_boost: float = 0.14
    keyword_weight_contains_boost: float = 0.06
    keyword_weight_token_boost: float = 0.04
    keyword_weight_anchor_max_df: int = 500
    keyword_weight_anchor_boost: float = 0.18
    glossary_context_max_chars: int = 4000
    format_extra_retries: int = 2
    latin_ratio_threshold: float = 2.0


@dataclass
class GeneralConfig:
    log_level: str = "INFO"
    prompt_style: str = "default"
    language: str = "auto"
    source_language: str = "auto"
    target_language: str = "zh"
    mcm_output_language_suffix: str = "source"
    mcm_auto_export: bool = True
    log_file: str = "logs/app.log"
    long_text_chunking_enabled: bool = True
    long_text_chunk_threshold_chars: int = 4000
    long_text_chunk_target_chars: int = 8000
    long_text_disable_thinking: bool = True
    short_text_batch_enabled: bool = False
    short_text_batch_max_chars: int = 50
    short_text_batch_size: int = 8


@dataclass
class PathsConfig:
    glossary_file: str = "glossary/glossary.json"
    vector_index_file: str = "glossary/vector_index.npy"


@dataclass
class ThreadsConfig:
    translation: int = 8
    vectorization: int = 8


@dataclass
class CacheConfig:
    translation_cache_size: int = 50000
    embedding_cache_size: int = 5000
    cache_persist_dir: str = "cache"
    cache_ttl_hours: float = 0  # 0 = no expiry


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_search: LLMConfig = field(default_factory=LLMConfig)
    llm_search_fallback: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    threads: ThreadsConfig = field(default_factory=ThreadsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


# Map section names to their dataclass types
_SECTION_MAP: dict[str, type] = {
    "llm": LLMConfig,
    "llm_search": LLMConfig,
    "llm_search_fallback": LLMConfig,
    "embedding": EmbeddingConfig,
    "rag": RAGConfig,
    "general": GeneralConfig,
    "paths": PathsConfig,
    "threads": ThreadsConfig,
    "cache": CacheConfig,
}


def validate_config(raw: dict) -> list[str]:
    """Validate a raw config dict against the schema. Returns list of error strings."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ["Config must be a JSON object"]

    for section_name, dc_type in _SECTION_MAP.items():
        section = raw.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            errors.append(f"'{section_name}' must be an object")
            continue

        dc_fields = {f.name: f for f in fields(dc_type)}
        for key, value in section.items():
            if key not in dc_fields:
                continue  # extra keys are allowed for forward compat
            f = dc_fields[key]
            expected = f.type
            # Basic type checks for primitives
            if expected == "int" and not isinstance(value, int) and value is not None:
                errors.append(f"'{section_name}.{key}' should be int, got {type(value).__name__}")
            elif expected == "float" and not isinstance(value, (int, float)) and value is not None:
                errors.append(f"'{section_name}.{key}' should be float, got {type(value).__name__}")
            elif expected == "str" and not isinstance(value, str):
                errors.append(f"'{section_name}.{key}' should be str, got {type(value).__name__}")
            elif expected == "bool" and not isinstance(value, bool) and value is not None:
                errors.append(f"'{section_name}.{key}' should be bool, got {type(value).__name__}")

    return errors


def _dict_to_dataclass(dc_type: type, data: dict) -> Any:
    """Convert a dict to a dataclass, ignoring unknown keys."""
    if not isinstance(data, dict):
        return dc_type()
    known = {f.name for f in fields(dc_type)}
    filtered = {k: v for k, v in data.items() if k in known}
    return dc_type(**filtered)


def config_to_dataclass(raw: dict) -> AppConfig:
    """Convert a raw JSON config dict to a typed AppConfig."""
    return AppConfig(
        llm=_dict_to_dataclass(LLMConfig, raw.get("llm", {})),
        llm_search=_dict_to_dataclass(LLMConfig, raw.get("llm_search", {})),
        llm_search_fallback=_dict_to_dataclass(LLMConfig, raw.get("llm_search_fallback", {})),
        embedding=_dict_to_dataclass(EmbeddingConfig, raw.get("embedding", {})),
        rag=_dict_to_dataclass(RAGConfig, raw.get("rag", {})),
        general=_dict_to_dataclass(GeneralConfig, raw.get("general", {})),
        paths=_dict_to_dataclass(PathsConfig, raw.get("paths", {})),
        threads=_dict_to_dataclass(ThreadsConfig, raw.get("threads", {})),
        cache=_dict_to_dataclass(CacheConfig, raw.get("cache", {})),
    )


def dataclass_to_dict(config: AppConfig) -> dict:
    """Convert an AppConfig back to a JSON-serializable dict."""
    return asdict(config)
