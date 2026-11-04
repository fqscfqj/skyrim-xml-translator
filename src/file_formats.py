"""Translation file type detection.

Two entry points, on purpose:

* :func:`detect_translation_file_type_from_extension` is the cheap GUI fast path
  (extension only, case-insensitive).
* :func:`detect_translation_file_type` also inspects content.  ``.txt`` files
  are only accepted as MCM when the sample actually looks like ``$key<TAB>value``
  lines (M-10), and XML-shaped content is classified with the same
  case-insensitive tag matching used for real ``.xml`` files (an ``<ESP>`` root
  is not required to be upper case).
"""

import codecs
import os
import re
from typing import Any, Optional


FILE_TYPE_XML = "xml"
FILE_TYPE_ESP_XML = "esp_xml"
FILE_TYPE_MCM = "mcm"
FILE_TYPE_RAW_PLUGIN = "raw_plugin"
FILE_TYPE_UNSUPPORTED = "unsupported"

RAW_PLUGIN_EXTENSIONS = frozenset({".esp", ".esm", ".esl"})

_CONTENT_SAMPLE_BYTES = 256 * 1024  # 只读头部判断内容形态，不整文件解析
_MCM_LINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")  # 与MCM处理器一致，不用splitlines
_MCM_MIN_RATIO = 0.5  # $key<TAB>value 行占比达到一半才判MCM


def _clean_path(file_path: str) -> str:
    return str(file_path or "").strip()  # 路径strip防空白


def _local_tag(tag: object) -> str:
    if not isinstance(tag, str):  # 注释/PI过滤
        return ""
    if "}" in tag:  # 剥离命名空间
        tag = tag.rsplit("}", 1)[-1]
    if ":" in tag:  # 剥离前缀
        tag = tag.rsplit(":", 1)[-1]
    return tag.strip()


def normalize_extension(file_path: str) -> str:
    return os.path.splitext(_clean_path(file_path))[1].lower()


def describe_extension(file_path: str) -> str:
    ext = normalize_extension(file_path)
    if ext:
        return ext
    try:  # 描述走i18n便于多语言
        from src.i18n import i18n as _i18n

        return _i18n.t("msg_no_extension", "(no extension)")
    except Exception:
        return "(no extension)"


def detect_translation_file_type_from_extension(file_path: str) -> str:
    """Extension-only fast path (case-insensitive), used by the GUI."""
    cleaned = _clean_path(file_path)  # strip后判扩展名
    ext = normalize_extension(cleaned)
    if ext == ".txt":
        return FILE_TYPE_MCM
    if ext == ".xml":
        return FILE_TYPE_XML
    if ext in RAW_PLUGIN_EXTENSIONS:
        return FILE_TYPE_RAW_PLUGIN
    return FILE_TYPE_UNSUPPORTED


def _decode_content_sample(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        even = raw[:len(raw) - (len(raw) % 2)]  # 采样可能截断半个字符
        try:
            return even.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if b"\x00" in raw:  # 无BOM的UTF-16：含TAB/换行才可信
        for encoding in ("utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "\t" in text or "\n" in text:
                return text
    for encoding in ("utf-8", "gbk", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_content_sample(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return _decode_content_sample(f.read(_CONTENT_SAMPLE_BYTES))


def _looks_like_xml_text(text: str) -> bool:
    return str(text or "").lstrip("\ufeff \t\r\n").startswith("<")


def _mcm_entry_ratio(text: str) -> tuple[float, int]:
    """Return ``(ratio, matched)`` of ``$key<TAB>value`` lines in *text*."""
    candidates = 0
    matched = 0
    for line in _MCM_LINE_SPLIT_RE.split(text or ""):
        if not line.strip() or line.lstrip().startswith(";"):  # 空行/注释不参评
            continue
        candidates += 1
        if "\t" not in line:
            continue
        key = line.split("\t", 1)[0].strip()
        if key.startswith("$") and len(key) > 1:  # MCM键形如 $OptionA
            matched += 1
    if not candidates:
        return 0.0, 0
    return matched / candidates, matched


def _looks_like_mcm_text(text: str) -> bool:
    ratio, matched = _mcm_entry_ratio(text)
    return matched >= 1 and ratio >= _MCM_MIN_RATIO


def _classify_parsed_xml(tree: Any) -> str:
    try:
        root = tree.getroot()
    except Exception:
        return FILE_TYPE_UNSUPPORTED
    has_string = has_source = has_esp = has_original = has_traduit = False
    count = 0
    try:
        for node in root.iter():
            name = _local_tag(getattr(node, "tag", "")).lower()
            if name == "string":
                has_string = True
                count += 1
            elif name == "source":
                has_source = True
            elif name == "esp":
                has_esp = True
                count += 1
            elif name == "original":
                has_original = True
            elif name == "traduit":
                has_traduit = True
            if has_esp and has_original and has_traduit:
                return FILE_TYPE_ESP_XML  # 三者齐备即ESP-ESM Translator格式
            if has_string and has_source:
                return FILE_TYPE_XML
    except Exception:
        return FILE_TYPE_UNSUPPORTED
    if has_esp and has_original:  # ESP+ORIGINAL即esp_xml
        return FILE_TYPE_ESP_XML
    if has_string and has_source:
        return FILE_TYPE_XML
    return FILE_TYPE_UNSUPPORTED  # 0条/无String判unsupported


def detect_translation_file_type(file_path: str) -> str:
    """Detect the type from extension *and* content.

    Raw plugins are never read.  ``.txt`` needs MCM-shaped content; XML-shaped
    content (any extension, including no extension) is parsed and classified by
    case-insensitive tag names.  ``UnsafeXMLDeclarationError`` is re-raised so
    callers can show why a file was refused.
    """
    from src.safe_xml import UnsafeXMLDeclarationError, parse_xml_file

    cleaned = _clean_path(file_path)
    if not cleaned:
        return FILE_TYPE_UNSUPPORTED
    ext = normalize_extension(cleaned)
    if ext in RAW_PLUGIN_EXTENSIONS:
        return FILE_TYPE_RAW_PLUGIN  # 不读二进制插件
    try:
        sample_text = _read_content_sample(cleaned)
    except OSError:
        return FILE_TYPE_UNSUPPORTED

    if ext == ".txt":
        return FILE_TYPE_MCM if _looks_like_mcm_text(sample_text) else FILE_TYPE_UNSUPPORTED
    if ext == ".xml" or _looks_like_xml_text(sample_text):
        try:
            tree = parse_xml_file(cleaned)  # UnsafeXML直接抛给调用方
        except UnsafeXMLDeclarationError:
            raise
        except Exception:
            return FILE_TYPE_UNSUPPORTED
        return _classify_parsed_xml(tree)
    if _looks_like_mcm_text(sample_text):
        return FILE_TYPE_MCM  # 无扩展名/其它扩展名但内容像MCM
    return FILE_TYPE_UNSUPPORTED


def classify_translation_file(file_path: str) -> tuple[str, Optional[str]]:
    # 返回(类型,原因)，Unsafe原因透出便于展示
    from src.safe_xml import UnsafeXMLDeclarationError

    try:
        return detect_translation_file_type(file_path), None
    except UnsafeXMLDeclarationError as e:
        return FILE_TYPE_UNSUPPORTED, str(e)
