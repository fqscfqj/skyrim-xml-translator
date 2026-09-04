"""Typed configuration schema for Skyrim XML Translator."""

from dataclasses import dataclass, field, fields, asdict
from typing import Any


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    api_mode: str = "chat_completions"
    max_retries: int = 3
    backoff_base: float = 0.5
    stream: bool = False
    json_response_format_enabled: bool = True
    request_timeout: int = 120
    request_timeout_step: int = 15
    request_timeout_max: int = 180
    retry_total_timeout: int = 300
    parameters: dict = field(default_factory=lambda: {
        "temperature": None,
        "top_p": None,
        "reasoning_protocol": "auto",
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
    vector_index_checkpoint_terms: int = 1000
    glossary_import_max_rows: int = 0
    glossary_import_max_field_chars: int = 0


@dataclass
class GeneralConfig:
    log_level: str = "INFO"
    prompt_style: str = "default"
    style_profile: str = "auto"
    language: str = "auto"
    color_mode: str = "auto"
    source_language: str = "auto"
    target_language: str = "zh"
    mcm_output_language_suffix: str = "source"
    mcm_auto_export: bool = True
    task_completion_sound_enabled: bool = False
    log_file: str = "logs/app.log"
    long_text_chunking_enabled: bool = True
    long_text_chunk_threshold_chars: int = 4000
    long_text_chunk_target_chars: int = 1800
    prompt_cache_warmup_enabled: bool = True
    short_text_batch_enabled: bool = False
    short_text_batch_max_chars: int = 50
    short_text_batch_size: int = 8
    short_text_batch_circuit_min_items: int = 16
    short_text_batch_circuit_fallback_ratio: float = 0.5


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
    embedding_cache_memory_mb: int = 256
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

_PARAM_ALLOW_NONE = {"temperature", "top_p", "enable_thinking", "reasoning_effort"}

_PARAM_CHOICES: dict[str, set] = {
    "reasoning_protocol": {"auto", "standard", "deepseek", "qwen",
                           "openrouter", "anthropic_adaptive", "gemini"},
    "reasoning_effort": {"minimal", "low", "medium", "high", "xhigh", "max"},
}

_CHOICES: dict[tuple[str, str], set] = {
    ("llm", "api_mode"): {"chat_completions", "responses"},
    ("llm_search", "api_mode"): {"chat_completions", "responses"},
    ("llm_search_fallback", "api_mode"): {"chat_completions", "responses"},
    ("general", "log_level"): {"DEBUG", "INFO", "WARNING", "ERROR"},
    ("general", "color_mode"): {"auto", "light", "dark"},
    ("general", "language"): {"auto", "en", "zh", "zh-Hant", "zh_Hant",
                              "ja", "ko", "fr", "de", "es", "ru"},
}

_RANGES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("llm", "max_retries"): (0, 10),
    ("llm", "backoff_base"): (0, 60),
    ("llm", "request_timeout"): (1, 3600),
    ("llm", "request_timeout_step"): (0, 3600),
    ("llm", "request_timeout_max"): (1, 3600),
    ("llm", "retry_total_timeout"): (1, 3600),
    ("embedding", "dimensions"): (1, 8192),
    ("rag", "similarity_threshold"): (0, 1),
    ("rag", "short_term_max_results"): (0, 100),
    ("rag", "long_term_max_results"): (0, 100),
    ("rag", "short_term_max_chars"): (1, 10000),
    ("rag", "keyword_max_queries"): (1, 1000),
    ("rag", "min_vector_score"): (0, 1),
    ("rag", "keyword_weight_candidate_pool_size"): (1, 1000),
    ("rag", "keyword_weight_keep_k"): (1, 1000),
    ("rag", "keyword_weight_min_primary_hits"): (0, 1000),
    ("rag", "keyword_weight_exact_boost"): (0, 10),
    ("rag", "keyword_weight_contains_boost"): (0, 10),
    ("rag", "keyword_weight_token_boost"): (0, 10),
    ("rag", "keyword_weight_anchor_max_df"): (1, 1000000),
    ("rag", "keyword_weight_anchor_boost"): (0, 10),
    ("rag", "glossary_context_max_chars"): (0, 100000),
    ("rag", "format_extra_retries"): (0, 10),
    ("rag", "latin_ratio_threshold"): (0, 100),
    ("rag", "vector_index_checkpoint_terms"): (1, 100000),
    ("rag", "glossary_import_max_rows"): (0, 1000000),
    ("rag", "glossary_import_max_field_chars"): (0, 100000),
    ("general", "long_text_chunk_threshold_chars"): (1, 100000),
    ("general", "long_text_chunk_target_chars"): (1, 100000),
    ("general", "short_text_batch_max_chars"): (1, 4000),
    ("general", "short_text_batch_size"): (1, 64),
    ("general", "short_text_batch_circuit_min_items"): (2, 10000),
    ("general", "short_text_batch_circuit_fallback_ratio"): (0.1, 1.0),
    ("threads", "translation"): (1, 64),
    ("threads", "vectorization"): (1, 64),
    ("cache", "translation_cache_size"): (1, 500000),
    ("cache", "embedding_cache_size"): (1, 50000),
    ("cache", "embedding_cache_memory_mb"): (32, 4096),
    ("cache", "cache_ttl_hours"): (0, 8760),
}


def _check_range(section: str, key: str, value: float,
                 errors: list[str]) -> None:
    bounds = _RANGES.get((section, key))
    if bounds is None or not isinstance(value, (int, float)):
        return
    lo, hi = bounds
    if isinstance(value, bool):
        return
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        errors.append(f"'{section}.{key}'={value!r} out of range [{lo}, {hi}]")


def _validate_parameters(section: str, params: object,
                         errors: list[str]) -> None:
    if params is None:
        return
    if not isinstance(params, dict):
        errors.append(f"'{section}.parameters' must be an object")
        return
    allowed = {"temperature", "top_p", "reasoning_protocol",
               "enable_thinking", "reasoning_effort"}
    for k, v in params.items():
        if k not in allowed:
            try:
                from src.logging_helper import emit as _emit
                _emit(None, None, "WARNING", f"Unknown {section}.parameters.{k} ignored",
                       module="config_schema", func="validate_config")
            except Exception:
                pass
            continue
        if v is None:
            if k not in _PARAM_ALLOW_NONE:
                errors.append(f"'{section}.parameters.{k}' must not be None")
            continue
        if k == "temperature":
            if type(v) not in (int, float) or isinstance(v, bool):
                errors.append(f"'{section}.parameters.temperature' should be float")
            elif not 0 <= float(v) <= 2:
                errors.append(f"'{section}.parameters.temperature'={v!r} out of range [0, 2]")
        elif k == "top_p":
            if type(v) not in (int, float) or isinstance(v, bool):
                errors.append(f"'{section}.parameters.top_p' should be float")
            elif not 0 <= float(v) <= 1:
                errors.append(f"'{section}.parameters.top_p'={v!r} out of range [0, 1]")
        elif k == "reasoning_protocol":
            if not isinstance(v, str) or v not in _PARAM_CHOICES["reasoning_protocol"]:
                errors.append(f"'{section}.parameters.reasoning_protocol' invalid: {v!r}")
        elif k == "enable_thinking":
            if type(v) is not bool:
                errors.append(f"'{section}.parameters.enable_thinking' should be bool")
        elif k == "reasoning_effort":
            if not isinstance(v, str) or v not in _PARAM_CHOICES["reasoning_effort"]:
                errors.append(f"'{section}.parameters.reasoning_effort' invalid: {v!r}")


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
                try:
                    from src.logging_helper import emit as _emit
                    _emit(None, None, "WARNING", f"Unknown config key '{section_name}.{key}' ignored",
                           module="config_schema", func="validate_config")
                except Exception:
                    pass
                continue
            if key == "parameters" and section_name in ("llm", "llm_search", "llm_search_fallback"):
                _validate_parameters(section_name, value, errors)
                continue
            f = dc_fields[key]
            expected = f.type
            if expected is int:
                if type(value) is not int:
                    errors.append(f"'{section_name}.{key}' should be int, got {type(value).__name__}")
                else:
                    _check_range(section_name, key, value, errors)
            elif expected is float:
                if type(value) not in (int, float) or isinstance(value, bool):
                    if value is None:
                        errors.append(f"'{section_name}.{key}' must not be None")
                    else:
                        errors.append(f"'{section_name}.{key}' should be float, got {type(value).__name__}")
                else:
                    _check_range(section_name, key, float(value), errors)
            elif expected is str:
                if not isinstance(value, str):
                    errors.append(f"'{section_name}.{key}' should be str, got {type(value).__name__}")
                elif (section_name, key) in _CHOICES and value not in _CHOICES[(section_name, key)]:
                    norm = str(value).strip()
                    if norm not in _CHOICES[(section_name, key)]:
                        errors.append(f"'{section_name}.{key}' invalid: {value!r}")
            elif expected is bool:
                if type(value) is not bool:
                    if value is None:
                        errors.append(f"'{section_name}.{key}' must not be None")
                    else:
                        errors.append(f"'{section_name}.{key}' should be bool, got {type(value).__name__}")
            elif expected is dict:
                if not isinstance(value, dict):
                    errors.append(f"'{section_name}.{key}' must be an object")

    try:
        general = raw.get("general") or {}
        if isinstance(general, dict):
            tgt = general.get("long_text_chunk_target_chars")
            thr = general.get("long_text_chunk_threshold_chars")
            if type(tgt) is int and type(thr) is int and tgt > thr:
                errors.append(
                    f"'general.long_text_chunk_target_chars' ({tgt}) exceeds "
                    f"'general.long_text_chunk_threshold_chars' ({thr})")
    except Exception:
        pass

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
