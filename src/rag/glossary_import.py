"""Unified glossary import parser (JSONv2 + legacy CSV/TSV compatible).

Design goals (see session plan):
- New primary format: structured JSON v1 envelope with rich optional fields.
- Old two-column CSV without header keeps working unchanged.
- New delimited variant: header-aware CSV/TSV with auto delimiter detection,
  multi-column rich fields, unknown-column tolerance.
- Encoding: UTF-8 with BOM preferred (utf-8-sig). GBK/non-UTF8 files raise
  an explicit error with line hints instead of being silently dropped.
- Limits reuse existing config semantics:
  rag.glossary_import_max_rows / rag.glossary_import_max_field_chars.
- Vectorization still uses only the term surface; rich fields are stored in
  a sidecar file and are read-only for retrieval in v1.

This module has no dependency on GUI / VectorStore / RAGEngine so it can be
unit-tested in isolation.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Import package format version. Bumped only on breaking envelope changes.
# Stored in vector_index.meta.json and translation policy fingerprint.
IMPORT_FORMAT_VERSION = 1
IMPORT_FORMAT_VERSION_STR = "glossary_import_v1"

_SAMPLE_INVALID_KEEP = 5


class GlossaryImportError(Exception):
    """Raised when an import file cannot be decoded or parsed."""


@dataclass
class ParsedGlossaryImport:
    terms: dict[str, str] = field(default_factory=dict)
    rich_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    deletes: list[str] = field(default_factory=list)
    total_rows: int = 0
    invalid_rows: int = 0
    limited_rows: int = 0
    duplicate_overwrites: int = 0
    unknown_fields: int = 0
    format_kind: str = ""
    encoding: str = "utf-8-sig"
    delimiter: str = ","
    source_file: str = ""
    samples_invalid: list[str] = field(default_factory=list)
    # Envelope-level provenance (JSONv2 only: source/created_at/format_version).
    # Never stored per-term; callers merge it into vector meta import_source.
    envelope: dict[str, Any] = field(default_factory=dict)

    @property
    def imported_terms(self) -> int:
        return len(self.terms)


# --- Header alias tables (lower-cased) ---

_TERM_KEYS = {
    "term", "term_text", "source_text", "en", "english", "原文", "英文",
}
_TRANSLATION_KEYS = {
    "translation", "translated", "zh", "chinese", "target", "dest",
    "destination", "译文", "中文", "翻译",
}
_DOMAIN_KEYS = {"domain", "领域", "domain_tag"}
_POS_KEYS = {"pos", "part_of_speech", "词性", "word_class"}
_PRIORITY_KEYS = {"priority", "weight", "优先级", "权重"}
_FORBIDDEN_KEYS = {"forbidden", "forbidden_translations", "banned", "禁用", "禁用译法"}
_EXAMPLES_KEYS = {"examples", "example", "context", "context_example", "例句", "上下文", "语境"}
_VARIANTS_KEYS = {"variants", "variant", "alt_translations", "多译法", "变体"}
_META_SOURCE_KEYS = {"meta_source", "origin", "dict_source", "来源", "出处", "数据来源"}
_CONFIDENCE_KEYS = {"confidence", "置信度", "score"}
_UPDATED_AT_KEYS = {"updated_at", "updated", "update_time", "更新时间"}
_NOTE_KEYS = {"note", "notes", "comment", "remark", "备注", "说明"}
_OP_KEYS = {"op", "operation", "action", "操作"}

# JSON entry may use "source" as the term surface; keep it after source_text/en.
_JSON_TERM_KEYS_ORDER = ("term", "source_text", "en", "source", "原文", "英文")
_JSON_TRANSLATION_KEYS_ORDER = ("translation", "zh", "target", "译文", "中文", "翻译")

_KNOWN_JSON_KEYS = (
    _TERM_KEYS | _TRANSLATION_KEYS | _DOMAIN_KEYS | _POS_KEYS | _PRIORITY_KEYS
    | _FORBIDDEN_KEYS | _EXAMPLES_KEYS | _VARIANTS_KEYS | _META_SOURCE_KEYS
    | _CONFIDENCE_KEYS | _UPDATED_AT_KEYS | _NOTE_KEYS | _OP_KEYS
    | {"term_text", "translated", "dest", "destination", "chinese",
       "source", "meta_source", "origin", "dict_source",
       "part_of_speech", "word_class", "weight", "banned",
       "forbidden_translations", "alt_translations", "example", "examples",
       "context", "context_example", "variant", "variants", "confidence",
       "score", "updated", "updated_at", "update_time", "note", "notes",
       "comment", "remark", "op", "operation", "action"}
)

_DELETE_OPS = {"delete", "del", "remove", "删除"}

_LIST_SPLIT_RE = re.compile(r"[;|｜；\n]+")


def _split_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in _LIST_SPLIT_RE.split(text) if part.strip()]


def _coerce_priority(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def _coerce_confidence(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        result = float(str(value).strip())
        if 0.0 <= result <= 1.0:
            return result
        return None
    except Exception:
        return None


def _normalize_header_cell(cell: Any) -> str:
    return str(cell or "").strip().lower()


def _build_header_map(header: list[str]) -> tuple[dict[str, int], int]:
    """Map known logical fields to column indexes. Returns (mapping, unknown)."""
    lowered = [_normalize_header_cell(cell) for cell in header]
    mapping: dict[str, int] = {}
    unknown = 0

    def claim(keys: set[str], logical: str) -> None:
        for idx, name in enumerate(lowered):
            if name in keys and logical not in mapping:
                mapping[logical] = idx
                break

    claim(_TERM_KEYS, "term")
    claim(_TRANSLATION_KEYS, "translation")
    claim(_DOMAIN_KEYS, "domain")
    claim(_POS_KEYS, "pos")
    claim(_PRIORITY_KEYS, "priority")
    claim(_FORBIDDEN_KEYS, "forbidden")
    claim(_EXAMPLES_KEYS, "examples")
    claim(_VARIANTS_KEYS, "variants")
    claim(_META_SOURCE_KEYS, "meta_source")
    claim(_CONFIDENCE_KEYS, "confidence")
    claim(_UPDATED_AT_KEYS, "updated_at")
    claim(_NOTE_KEYS, "note")
    claim(_OP_KEYS, "op")

    known_names = (
        _TERM_KEYS | _TRANSLATION_KEYS | _DOMAIN_KEYS | _POS_KEYS
        | _PRIORITY_KEYS | _FORBIDDEN_KEYS | _EXAMPLES_KEYS | _VARIANTS_KEYS
        | _META_SOURCE_KEYS | _CONFIDENCE_KEYS | _UPDATED_AT_KEYS
        | _NOTE_KEYS | _OP_KEYS
    )
    for name in lowered:
        if name and name not in known_names:
            unknown += 1
    return mapping, unknown


def _looks_like_header(row: list[str]) -> bool:
    lowered = {_normalize_header_cell(cell) for cell in row}
    has_term = bool(lowered & _TERM_KEYS)
    has_translation = bool(lowered & _TRANSLATION_KEYS)
    if not (has_term and has_translation):
        return False
    # "Term,Translation" as a data row is the canonical legacy fixture.
    # Only treat the first row as a header when it declares at least one
    # extra known column (domain/pos/priority/...) or unknown column.
    known_names = (
        _DOMAIN_KEYS | _POS_KEYS | _PRIORITY_KEYS | _FORBIDDEN_KEYS
        | _EXAMPLES_KEYS | _VARIANTS_KEYS | _META_SOURCE_KEYS
        | _CONFIDENCE_KEYS | _UPDATED_AT_KEYS | _NOTE_KEYS | _OP_KEYS
    )
    if len(row) > 2:
        return True
    return any(name in known_names or name not in (_TERM_KEYS | _TRANSLATION_KEYS) for name in lowered if name)


def _detect_delimiter(sample_text: str, extension: str) -> str:
    if extension == ".tsv":
        return "\t"
    if extension == ".csv":
        # CSV keeps comma even if content contains tabs; sniff only for ';' edge.
        if ";" in sample_text and sample_text.count(";") > sample_text.count(","):
            return ";"
        return ","
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=[",", "\t", ";"])
        if dialect.delimiter in (",", "\t", ";"):
            return dialect.delimiter
    except Exception:
        pass
    tab_count = sample_text.count("\t")
    comma_count = sample_text.count(",")
    semi_count = sample_text.count(";")
    if tab_count > comma_count and tab_count >= semi_count:
        return "\t"
    if semi_count > comma_count and semi_count > tab_count:
        return ";"
    return ","


def _read_text_with_encoding(file_path: str) -> tuple[str, str]:
    """Read full text preferring utf-8-sig; raise explicit error otherwise."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            return handle.read(), "utf-8-sig"
    except UnicodeDecodeError as exc:
        reason = (
            f"无法以 UTF-8 解码导入文件 '{file_path}'"
            f"（字节 {exc.start}-{exc.end}，原因：{exc.reason}）。"
            "请将其另存为 UTF-8 编码（含/不含 BOM 均可）后重试；"
            "GBK/ANSI 编码需先转换，直接导入会被拒绝以避免乱码入库。"
        )
        raise GlossaryImportError(reason) from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise GlossaryImportError(f"无法读取导入文件 '{file_path}'：{exc}") from exc


def _record_invalid(result: ParsedGlossaryImport, message: str) -> None:
    result.invalid_rows += 1
    if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
        result.samples_invalid.append(message)


def _apply_entry(
    result: ParsedGlossaryImport,
    term: str,
    translation: str,
    rich: dict[str, Any],
    raw_op: str = "",
    *,
    max_field_chars: int,
    line_hint: str = "",
) -> None:
    term = (term or "").strip()
    translation = (translation or "").strip()
    op = (raw_op or "").strip().lower()
    if op in _DELETE_OPS:
        if not term:
            _record_invalid(result, f"{line_hint}删除行缺少词面，已跳过")
            return
        if term in result.terms:
            del result.terms[term]
            result.rich_meta.pop(term, None)
            result.duplicate_overwrites += 1
        if term not in result.deletes:
            result.deletes.append(term)
        else:
            result.duplicate_overwrites += 1
        return
    if not term or not translation:
        _record_invalid(result, f"{line_hint}空词面或空译文，已跳过")
        return
    if max_field_chars and (len(term) > max_field_chars or len(translation) > max_field_chars):
        result.limited_rows += 1
        return
    if term in result.terms:
        result.duplicate_overwrites += 1
        # A later upsert cancels an earlier delete declaration for the same term.
        if term in result.deletes:
            result.deletes.remove(term)
    result.terms[term] = translation
    cleaned = {key: value for key, value in rich.items() if value not in (None, "", [], {})}
    if cleaned:
        result.rich_meta[term] = cleaned


def parse_delimited_text(
    text: str,
    *,
    delimiter: str,
    max_rows: int = 0,
    max_field_chars: int = 0,
    source_file: str = "",
    progress_callback: Optional[Callable[[int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ParsedGlossaryImport:
    import io

    result = ParsedGlossaryImport(
        source_file=source_file,
        encoding="utf-8-sig",
        delimiter=delimiter,
    )
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        result.format_kind = "csv_legacy"
        return result

    header_map: dict[str, int] = {}
    data_start = 0
    if _looks_like_header(rows[0]):
        header_map, unknown = _build_header_map(rows[0])
        result.unknown_fields += unknown
        data_start = 1
        result.format_kind = "tsv_header" if delimiter == "\t" else "csv_header"
    else:
        result.format_kind = "tsv_legacy" if delimiter == "\t" else "csv_legacy"

    data_rows = rows[data_start:]
    total = len(data_rows)
    for data_index, row in enumerate(data_rows, start=1):
        if should_stop is not None:
            try:
                if should_stop():
                    break
            except Exception:
                pass
        if max_rows and data_index > max_rows:
            result.limited_rows += 1
            continue
        result.total_rows += 1
        line_hint = f"第{data_index + data_start}行:"
        if header_map:
            def col(name: str) -> str:
                idx = header_map.get(name, -1)
                if idx is None or idx < 0 or idx >= len(row):
                    return ""
                return str(row[idx] or "").strip()

            term = col("term")
            translation = col("translation")
            rich: dict[str, Any] = {}
            for key in ("domain", "pos", "meta_source", "updated_at", "note"):
                value = col(key)
                if value:
                    rich[key] = value
            priority = _coerce_priority(col("priority"))
            if priority is not None:
                rich["priority"] = priority
            confidence = _coerce_confidence(col("confidence"))
            if confidence is not None:
                rich["confidence"] = confidence
            for key in ("forbidden", "examples", "variants"):
                items = _split_list_field(col(key))
                if items:
                    rich[key] = items
            _apply_entry(
                result, term, translation, rich, col("op"),
                max_field_chars=max_field_chars, line_hint=line_hint,
            )
        else:
            if len(row) < 2:
                _record_invalid(result, f"{line_hint}列数不足 2 列，已跳过")
                continue
            _apply_entry(
                result, str(row[0] or ""), str(row[1] or ""), {},
                max_field_chars=max_field_chars, line_hint=line_hint,
            )
        if progress_callback is not None and total > 0 and (data_index % 500 == 0 or data_index == total):
            try:
                progress_callback(int(data_index / total * 100))
            except Exception:
                pass
    return result


def _entry_from_json_object(obj: Any) -> tuple[str, str, dict[str, Any], str, int]:
    """Return (term, translation, rich, op, unknown_count)."""
    if not isinstance(obj, dict):
        return "", "", {}, "", 0
    unknown = sum(1 for key in obj.keys() if str(key).strip().lower() not in _KNOWN_JSON_KEYS)

    def pick(keys: tuple[str, ...]) -> str:
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                value = obj[key]
                if isinstance(value, (list, tuple)):
                    return str(value[0] if value else "").strip()
                return str(value or "").strip()
        # case-insensitive fallback
        lowered = {str(key).strip().lower(): value for key, value in obj.items()}
        for key in keys:
            if key in lowered and lowered[key] not in (None, ""):
                value = lowered[key]
                if isinstance(value, (list, tuple)):
                    return str(value[0] if value else "").strip()
                return str(value or "").strip()
        return ""

    def pick_raw(keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in obj:
                return obj[key]
        lowered = {str(key).strip().lower(): value for key, value in obj.items()}
        for key in keys:
            if key in lowered:
                return lowered[key]
        return None

    term = pick(_JSON_TERM_KEYS_ORDER)
    if not term:
        term = pick(tuple(sorted(_TERM_KEYS)))
    translation = pick(_JSON_TRANSLATION_KEYS_ORDER)
    if not translation:
        translation = pick(tuple(sorted(_TRANSLATION_KEYS)))
    op = pick(tuple(sorted(_OP_KEYS)))

    rich: dict[str, Any] = {}
    domain = pick(tuple(sorted(_DOMAIN_KEYS)))
    if domain:
        rich["domain"] = domain
    pos = pick(tuple(sorted(_POS_KEYS)))
    if pos:
        rich["pos"] = pos
    meta_source = pick(tuple(sorted(_META_SOURCE_KEYS)))
    if meta_source:
        rich["source"] = meta_source
    updated_at = pick(tuple(sorted(_UPDATED_AT_KEYS)))
    if updated_at:
        rich["updated_at"] = updated_at
    note = pick(tuple(sorted(_NOTE_KEYS)))
    if note:
        rich["note"] = note
    priority = _coerce_priority(pick_raw(tuple(sorted(_PRIORITY_KEYS))))
    if priority is not None:
        rich["priority"] = priority
    confidence = _coerce_confidence(pick_raw(tuple(sorted(_CONFIDENCE_KEYS))))
    if confidence is not None:
        rich["confidence"] = confidence
    for logical, keys in (
        ("forbidden", tuple(sorted(_FORBIDDEN_KEYS))),
        ("examples", tuple(sorted(_EXAMPLES_KEYS))),
        ("variants", tuple(sorted(_VARIANTS_KEYS))),
    ):
        raw = pick_raw(keys)
        items = _split_list_field(raw)
        if items:
            rich[logical] = items
    return term, translation, rich, op, unknown


def parse_json_text(
    text: str,
    *,
    max_rows: int = 0,
    max_field_chars: int = 0,
    source_file: str = "",
    progress_callback: Optional[Callable[[int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ParsedGlossaryImport:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GlossaryImportError(
            f"导入 JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
        ) from exc

    result = ParsedGlossaryImport(
        source_file=source_file, encoding="utf-8-sig",
        delimiter="", format_kind="json_v2",
    )
    envelope_source = ""
    if isinstance(payload, dict) and "terms" in payload:
        terms_raw = payload.get("terms")
        envelope_source = str(payload.get("source") or "").strip()
        created_at = payload.get("created_at")
        format_version = payload.get("format_version")
        envelope: dict[str, Any] = {}
        if envelope_source:
            envelope["source"] = envelope_source
        if created_at not in (None, ""):
            envelope["created_at"] = str(created_at)
        if format_version not in (None, ""):
            try:
                envelope["format_version"] = int(format_version)  # type: ignore[arg-type]
            except Exception:
                envelope["format_version"] = format_version
        if envelope:
            result.envelope = envelope
    elif isinstance(payload, dict) and "format_version" in payload and "terms" not in payload:
        raise GlossaryImportError("导入 JSON 包含 format_version 但缺少 terms 数组/对象")
    else:
        terms_raw = payload

    entries: list[Any] = []
    if isinstance(terms_raw, dict):
        for term_key, value in terms_raw.items():
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("term", term_key)
                entries.append(merged)
            elif isinstance(value, (list, tuple)) and len(value) >= 1:
                entries.append({"term": term_key, "translation": str(value[0] or "")})
            else:
                entries.append({"term": str(term_key), "translation": str(value or "")})
        if not isinstance(payload, dict) or "terms" not in payload:
            result.format_kind = "json_legacy"
    elif isinstance(terms_raw, list):
        entries = list(terms_raw)
    else:
        raise GlossaryImportError("导入 JSON 的 terms 必须是数组或对象")

    total = len(entries)
    for data_index, entry in enumerate(entries, start=1):
        if should_stop is not None:
            try:
                if should_stop():
                    break
            except Exception:
                pass
        if max_rows and data_index > max_rows:
            result.limited_rows += 1
            continue
        result.total_rows += 1
        line_hint = f"第{data_index}条:"
        if isinstance(entry, (list, tuple)):
            term = str(entry[0] if len(entry) > 0 else "").strip()
            translation = str(entry[1] if len(entry) > 1 else "").strip()
            _apply_entry(result, term, translation, {}, max_field_chars=max_field_chars, line_hint=line_hint)
            continue
        term, translation, rich, op, unknown = _entry_from_json_object(entry)
        result.unknown_fields += unknown
        if envelope_source and "source" not in rich:
            rich["source"] = envelope_source
        _apply_entry(result, term, translation, rich, op, max_field_chars=max_field_chars, line_hint=line_hint)
        if progress_callback is not None and total > 0 and (data_index % 500 == 0 or data_index == total):
            try:
                progress_callback(int(data_index / total * 100))
            except Exception:
                pass
    # Defensive: never leak reserved "__...__" pseudo-terms into sidecar/vectors,
    # even if a source file literally contains such a key.
    for reserved in [key for key in result.terms if key.startswith("__") and key.endswith("__")]:
        result.terms.pop(reserved, None)
        result.rich_meta.pop(reserved, None)
    for reserved in [key for key in result.rich_meta if key.startswith("__") and key.endswith("__")]:
        result.rich_meta.pop(reserved, None)
    return result


def parse_glossary_file(
    file_path: str,
    *,
    max_rows: int = 0,
    max_field_chars: int = 0,
    progress_callback: Optional[Callable[[int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ParsedGlossaryImport:
    """Parse CSV/TSV/TXT/JSON glossary import files with a unified report."""
    max_rows = max(0, int(max_rows or 0))
    max_field_chars = max(0, int(max_field_chars or 0))
    extension = os.path.splitext(str(file_path or ""))[1].strip().lower()
    text, encoding = _read_text_with_encoding(file_path)
    if extension == ".json":
        result = parse_json_text(
            text, max_rows=max_rows, max_field_chars=max_field_chars,
            source_file=str(file_path), progress_callback=progress_callback,
            should_stop=should_stop,
        )
        result.encoding = encoding
        return result
    # Delimited path (csv/tsv/txt/unknown defaults to delimited sniffing).
    sample = text[:8192]
    delimiter = _detect_delimiter(sample, extension)
    result = parse_delimited_text(
        text, delimiter=delimiter, max_rows=max_rows,
        max_field_chars=max_field_chars, source_file=str(file_path),
        progress_callback=progress_callback, should_stop=should_stop,
    )
    result.encoding = encoding
    return result


def summarize_import(result: ParsedGlossaryImport) -> str:
    """One-line human-readable summary for logs and dialogs (no secrets)."""
    return (
        f"解析完成（{result.format_kind or 'unknown'}）："
        f"有效 {len(result.terms)} 条，删除 {len(result.deletes)} 条，"
        f"无效 {result.invalid_rows} 行，"
        f"受限 {result.limited_rows} 行，"
        f"文件内覆盖 {result.duplicate_overwrites} 次，"
        f"未知字段 {result.unknown_fields} 个。"
    )
