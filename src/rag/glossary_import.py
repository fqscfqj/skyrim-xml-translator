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

import codecs
import csv
import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Import package format version. Bumped only on breaking envelope changes.
# Stored in vector_index.meta.json and translation policy fingerprint.
IMPORT_FORMAT_VERSION = 1
IMPORT_FORMAT_VERSION_STR = "glossary_import_v1"

_SAMPLE_INVALID_KEEP = 5


class GlossaryImportError(FileNotFoundError):
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
    # Internal indexes keep duplicate/delete matching O(1) while preserving
    # the public terms/rich_meta dictionaries and their original spelling.
    _normalized_terms: dict[str, str] = field(default_factory=dict, repr=False)
    _normalized_deletes: set[str] = field(default_factory=set, repr=False)

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
_META_SOURCE_KEYS = {"meta_source", "origin", "dict_source", "source", "来源", "出处", "数据来源"}
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
# Explicit upsert whitelist (G-09).  Empty op is the legacy two-column default.
_UPSERT_OPS = {"", "upsert", "add", "update", "replace"}
# Candidate delimiters for trial parsing (P0-10).
_TRIAL_DELIMITERS = (",", "\t", ";", "|")
# Never let a legitimate long field explode as a raw csv.Error (P0-10/G-08).
_MIN_CSV_FIELD_SIZE_LIMIT = 1 << 20  # 1 MiB

_LIST_SPLIT_RE = re.compile(r"[;|｜；\n]+")
_TRANSLATION_SPLIT_RE = re.compile(r"[;|｜；\n]+")


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


def _split_translation_variants(value: Any) -> tuple[str, list[str]]:
    """Split ``主译;副译;副译`` into (main translation, extra variants) (R-30).

    Only explicit multi-value separators are split; commas are left alone
    because natural translations (e.g. Chinese lists) commonly contain them.
    """
    if value is None:
        return "", []
    if isinstance(value, (list, tuple)):
        parts = [str(item or "").strip() for item in value]
        parts = [part for part in parts if part]
    else:
        text = str(value or "").strip()
        if not text:
            return "", []
        parts = [part.strip() for part in _TRANSLATION_SPLIT_RE.split(text) if part.strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _coerce_priority(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(str(value).strip())
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
    claim(_META_SOURCE_KEYS, "source")
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


def _looks_like_header(row: list[str], following_rows: Optional[list[list[str]]] = None) -> bool:
    lowered = {_normalize_header_cell(cell) for cell in row}
    has_term = bool(lowered & _TERM_KEYS)
    has_translation = bool(lowered & _TRANSLATION_KEYS)
    if not (has_term and has_translation):
        return False
    # G-04: a first row that is exactly "<term key>,<translation key>"
    # (e.g. "Term,Translation" / "原文,译文") is a header, not a glossary entry.
    if len(row) == 2:
        if not (
            _normalize_header_cell(row[0]) in _TERM_KEYS
            and _normalize_header_cell(row[1]) in _TRANSLATION_KEYS
        ):
            return False
        # Keep the canonical legacy two-column fixture working: when no later
        # row has 2+ columns there is no evidence of a real table, so treat the
        # row as data.  Callers without lookahead context assume a header.
        if following_rows is None:
            return True
        return any(len(next_row) >= 2 for next_row in following_rows)
    return True


def _score_delimiter(sample_text: str, delimiter: str) -> float:
    """Fraction of non-empty rows that agree on a >=2-column width.

    Trial parsing (not raw character counts) prevents semicolons inside a
    translation from being mistaken for the column separator (P0-10).
    """
    if not sample_text or not delimiter:
        return 0.0
    try:
        rows = list(csv.reader(io.StringIO(sample_text), delimiter=delimiter))
    except csv.Error:
        return 0.0
    counts = [len(row) for row in rows if any(str(cell or "").strip() for cell in row)]
    if not counts:
        return 0.0
    multi = [count for count in counts if count >= 2]
    if not multi:
        return 0.0
    modal = Counter(multi).most_common(1)[0][0]
    return multi.count(modal) / float(len(counts))


def _default_delimiter(extension: str) -> str:
    return "\t" if extension == ".tsv" else ","


def _detect_delimiter(sample_text: str, extension: str) -> str:
    """Pick the delimiter whose trial parse yields the most consistent rows.

    Candidates are tried starting with the extension default, so ties (e.g.
    "Sword,剑;刀;刃;锋" parses to 2 columns with ',' and 4 with ';') keep the
    extension-based default.  When no candidate yields a >=2-column parse the
    default is kept as well.
    """
    default = _default_delimiter(extension)
    best = default
    best_score = -1.0
    ordered = [default] + [d for d in _TRIAL_DELIMITERS if d != default]
    for candidate in ordered:
        score = _score_delimiter(sample_text, candidate)
        if score > best_score + 1e-9:
            best = candidate
            best_score = score
    if best_score <= 0.0:
        return default
    return best


def _decode_utf16_without_bom(raw: bytes) -> Optional[tuple[str, str]]:
    """Decode BOM-less UTF-16 by picking the endianness with the NUL pattern."""
    even_nuls = raw[0::2].count(0)
    odd_nuls = raw[1::2].count(0)
    order = ("utf-16-le", "utf-16-be") if odd_nuls >= even_nuls else ("utf-16-be", "utf-16-le")
    for encoding in order:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        # A successful decode must not leave embedded NUL characters behind.
        if "\x00" in text:
            continue
        return text, encoding
    return None


def _read_text_with_encoding(file_path: str) -> tuple[str, str]:
    """Read full text, tolerating UTF-8 BOM and BOM-less UTF-16 (G-07).

    UTF-16 files (with or without BOM) and NUL-laden content used to be
    rejected outright; they are now re-decoded so a Notepad "Unicode" export
    imports instead of failing.  Genuinely non-UTF-8 single-byte encodings
    (GBK/ANSI) still raise an explicit error: silently guessing them produced
    mojibake, so the user is asked to convert instead.
    """
    try:
        with open(file_path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError as exc:
        raise GlossaryImportError(f"导入文件不存在 '{file_path}'：{exc}") from exc
    except OSError as exc:
        raise GlossaryImportError(f"无法读取导入文件 '{file_path}'：{exc}") from exc

    if not raw:
        return "", "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF8):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            _raise_decode_error(file_path, exc)
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError as exc:
            _raise_decode_error(file_path, exc)
    if b"\x00" in raw[:65536] or (
        len(raw) >= 4 and (raw[0::2].count(0) or raw[1::2].count(0))
    ):
        decoded = _decode_utf16_without_bom(raw)
        if decoded is not None:
            return decoded
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        _raise_decode_error(file_path, exc)


def _raise_decode_error(file_path: str, exc: UnicodeDecodeError):
    reason = (
        f"无法以 UTF-8 解码导入文件 '{file_path}'"
        f"（字节 {exc.start}-{exc.end}，原因：{exc.reason}）。"
        "请将其另存为 UTF-8 编码（含/不含 BOM 均可）后重试；"
        "GBK/ANSI 编码需先转换，直接导入会被拒绝以避免乱码入库。"
    )
    raise GlossaryImportError(reason) from exc


def _record_invalid(result: ParsedGlossaryImport, message: str) -> None:
    result.invalid_rows += 1
    if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
        result.samples_invalid.append(message)


def _record_limited(result: ParsedGlossaryImport, message: str) -> None:
    result.limited_rows += 1
    if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
        result.samples_invalid.append(message)


def _normalize_import_key(text: str) -> str:
    import unicodedata as _unicodedata
    try:
        normalized = _unicodedata.normalize("NFKC", str(text or ""))
    except Exception:
        normalized = str(text or "")
    return normalized.strip().lower()


def _find_term_key(
    mapping: dict[str, Any],
    term: str,
    normalized_index: Optional[dict[str, str]] = None,
) -> Optional[str]:
    if term in mapping:
        return term
    target = _normalize_import_key(term)
    if not target:
        return None
    if normalized_index is not None:
        indexed = normalized_index.get(target)
        if indexed is not None and indexed in mapping:
            return indexed
    for key in mapping.keys():
        if _normalize_import_key(key) == target:
            return key
    return None


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
    # R-30: multi-translation surfaces ("甲;乙;丙" or a JSON list) become the
    # primary translation plus explicit variants instead of dropping the rest.
    translation, derived_variants = _split_translation_variants(translation)
    if derived_variants:
        existing_variants = _split_list_field(rich.get("variants"))
        merged: list[str] = []
        for candidate in [*existing_variants, *derived_variants]:
            if candidate and candidate != translation and candidate not in merged:
                merged.append(candidate)
        if merged:
            rich["variants"] = merged
    if op and op not in _DELETE_OPS and op not in _UPSERT_OPS:
        # G-09: unknown operations used to be silently treated as upsert.
        _record_invalid(
            result,
            f"{line_hint}未知操作 op='{str(raw_op).strip()}'"
            "（支持 upsert/add/update/replace/delete/remove），已跳过",
        )
        return
    if op in _DELETE_OPS:
        if not term:
            _record_invalid(result, f"{line_hint}删除行缺少词面，已跳过")
            return
        norm = _normalize_import_key(term)
        existing = _find_term_key(result.terms, term, result._normalized_terms)
        if existing is not None:
            del result.terms[existing]
            result._normalized_terms.pop(norm, None)
            result.rich_meta.pop(existing, None)
            # Keep compatibility with manually supplied rich_meta aliases.
            for alias in [k for k in list(result.rich_meta.keys()) if _normalize_import_key(k) == norm]:
                result.rich_meta.pop(alias, None)
            result.duplicate_overwrites += 1
        if norm not in result._normalized_deletes:
            result.deletes.append(term)
            result._normalized_deletes.add(norm)
        else:
            result.duplicate_overwrites += 1
        return
    if not term or not translation:
        _record_invalid(result, f"{line_hint}空词面或空译文，已跳过")
        return
    if max_field_chars and (len(term) > max_field_chars or len(translation) > max_field_chars):
        _record_limited(result, f"{line_hint}字段超限（>{max_field_chars}字符）已截断：{term[:30]}")
        return
    norm = _normalize_import_key(term)
    existing = _find_term_key(result.terms, term, result._normalized_terms)
    if existing is not None:
        result.duplicate_overwrites += 1
        if norm in result._normalized_deletes:
            result.deletes[:] = [d for d in result.deletes if _normalize_import_key(d) != norm]
            result._normalized_deletes.discard(norm)
        if existing != term:
            result.terms.pop(existing, None)
            result._normalized_terms.pop(norm, None)
            result.rich_meta.pop(existing, None)
    result.terms[term] = translation
    result._normalized_terms[norm] = term
    cleaned = {key: value for key, value in rich.items() if value not in (None, "", [], {})}
    if max_field_chars:
        # G-08: rich fields must obey the same size cap as term/translation,
        # otherwise a 5000-char note slips past max_field_chars unnoticed.
        for key, value in list(cleaned.items()):
            too_long = (
                isinstance(value, str) and len(value) > max_field_chars
            ) or (
                isinstance(value, (list, tuple))
                and any(len(str(item)) > max_field_chars for item in value)
            )
            if too_long:
                cleaned.pop(key, None)
                _record_limited(
                    result,
                    f"{line_hint}富字段 '{key}' 超限（>{max_field_chars}字符）已丢弃：{term[:30]}",
                )
    if cleaned:
        result.rich_meta[term] = cleaned


def _ensure_csv_field_size_limit(max_field_chars: int = 0) -> None:
    """Raise the stdlib csv field cap to at least 1 MiB (P0-10/G-08)."""
    target = max(_MIN_CSV_FIELD_SIZE_LIMIT, int(max_field_chars or 0) + 1024)
    target = min(target, 2 ** 31 - 1)
    try:
        if csv.field_size_limit() < target:
            csv.field_size_limit(target)
    except (OverflowError, ValueError, csv.Error):
        try:
            csv.field_size_limit(_MIN_CSV_FIELD_SIZE_LIMIT)
        except (OverflowError, ValueError, csv.Error):
            pass


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
    _ensure_csv_field_size_limit(max_field_chars)

    result = ParsedGlossaryImport(
        source_file=source_file,
        encoding="utf-8-sig",
        delimiter=delimiter,
    )
    # strict=True turns malformed rows (unclosed quotes, stray quote inside an
    # unquoted field) into csv.Error instead of silently merging/dropping rows.
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
    try:
        rows = list(reader)
    except csv.Error as exc:
        line_no = max(1, int(getattr(reader, "line_num", 0) or 0))
        raise GlossaryImportError(
            f"CSV 解析失败（分隔符 {delimiter!r}，约第 {line_no} 行）：{exc}。"
            "请检查是否存在未闭合的引号或跨行字段，修正后重试。"
        ) from exc
    if not rows:
        result.format_kind = "csv_legacy"
        return result

    header_map: dict[str, int] = {}
    data_start = 0
    if _looks_like_header(rows[0], rows[1:]):
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
            remaining = total - data_index + 1
            result.limited_rows += remaining
            if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
                result.samples_invalid.append(f"第{data_index + data_start}行:超出 max_rows={max_rows}，剩余{remaining}行已截断")
            break
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
            for key in ("domain", "pos", "source", "updated_at", "note"):
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
            if len(row) > 2:
                result.unknown_fields += 1
                if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
                    result.samples_invalid.append(f"{line_hint}legacy 多出 {len(row) - 2} 列已忽略，仅取前两列")
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
    explicit_term = pick(("term", "source_text", "en", "原文", "英文"))
    if not term:
        term = pick(tuple(sorted(_TERM_KEYS)))
    translation = pick(_JSON_TRANSLATION_KEYS_ORDER)
    if not translation:
        translation = pick(tuple(sorted(_TRANSLATION_KEYS)))

    # R-30: a JSON translation list keeps the first entry as the main
    # translation and the rest as variants instead of dropping value[1:].
    extra_variants: list[str] = []
    raw_translation: Any = None
    lowered_keys = {str(key).strip().lower(): value for key, value in obj.items()}
    for key in _JSON_TRANSLATION_KEYS_ORDER:
        candidate = obj.get(key, lowered_keys.get(key))
        if isinstance(candidate, (list, tuple)) and candidate:
            raw_translation = candidate
            break
    if raw_translation is not None:
        parts = [str(item or "").strip() for item in raw_translation]
        parts = [part for part in parts if part]
        if parts:
            translation = parts[0]
            extra_variants = parts[1:]

    op = pick(tuple(sorted(_OP_KEYS)))

    rich: dict[str, Any] = {}
    domain = pick(tuple(sorted(_DOMAIN_KEYS)))
    if domain:
        rich["domain"] = domain
    pos = pick(tuple(sorted(_POS_KEYS)))
    if pos:
        rich["pos"] = pos
    meta_source = pick(tuple(sorted(_META_SOURCE_KEYS)))
    if meta_source and not explicit_term and _normalize_import_key(meta_source) == _normalize_import_key(term):
        # In a legacy entry `source` can be the term surface; do not also store
        # it as provenance in that case.
        meta_source = ""
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
    if extra_variants:
        merged = [v for v in rich.get("variants", []) if v]
        for variant in extra_variants:
            if variant and variant != translation and variant not in merged:
                merged.append(variant)
        if merged:
            rich["variants"] = merged
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
                # Keep the whole list: the first entry is the main translation,
                # the rest become variants (R-30).
                entries.append({"term": term_key,
                                "translation": [str(item or "") for item in value]})
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
            remaining = total - data_index + 1
            result.limited_rows += remaining
            if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
                result.samples_invalid.append(f"第{data_index}条:超出 max_rows={max_rows}，剩余{remaining}条已截断")
            break
        result.total_rows += 1
        line_hint = f"第{data_index}条:"
        if isinstance(entry, (list, tuple)):
            term = str(entry[0] if len(entry) > 0 else "").strip()
            translation = str(entry[1] if len(entry) > 1 else "").strip()
            if len(entry) > 2:
                result.unknown_fields += 1
                if len(result.samples_invalid) < _SAMPLE_INVALID_KEEP:
                    result.samples_invalid.append(f"{line_hint}legacy 多出 {len(entry) - 2} 列已忽略，仅取前两列")
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
    # Ensure delimiter trial parsing sees the same field cap as the real parse.
    _ensure_csv_field_size_limit(max_field_chars)
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
