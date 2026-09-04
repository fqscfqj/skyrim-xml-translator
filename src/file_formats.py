import os
from typing import Optional


FILE_TYPE_XML = "xml"
FILE_TYPE_ESP_XML = "esp_xml"
FILE_TYPE_MCM = "mcm"
FILE_TYPE_RAW_PLUGIN = "raw_plugin"
FILE_TYPE_UNSUPPORTED = "unsupported"

RAW_PLUGIN_EXTENSIONS = frozenset({".esp", ".esm", ".esl"})


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
    cleaned = _clean_path(file_path)  # strip后判扩展名
    ext = normalize_extension(cleaned)
    if ext == ".txt":
        return FILE_TYPE_MCM
    if ext == ".xml":
        return FILE_TYPE_XML
    if ext in RAW_PLUGIN_EXTENSIONS:
        return FILE_TYPE_RAW_PLUGIN
    return FILE_TYPE_UNSUPPORTED


def detect_translation_file_type(file_path: str) -> str:
    # 内容感知识别：0条/无String判unsupported，Unsafe原因透出
    from src.safe_xml import UnsafeXMLDeclarationError, parse_xml_file

    cleaned = _clean_path(file_path)
    if not cleaned:
        return FILE_TYPE_UNSUPPORTED
    ext_type = detect_translation_file_type_from_extension(cleaned)
    if ext_type != FILE_TYPE_XML:
        return ext_type
    try:
        tree = parse_xml_file(cleaned)  # UnsafeXML直接抛给调用方
    except UnsafeXMLDeclarationError:
        raise
    except Exception:
        return FILE_TYPE_UNSUPPORTED
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
                return FILE_TYPE_ESP_XML
            if has_string and has_source:
                return FILE_TYPE_XML
    except Exception:
        return FILE_TYPE_UNSUPPORTED
    if has_esp and has_original:  # ESP+ORIGINAL即esp_xml
        return FILE_TYPE_ESP_XML
    if has_string and has_source:
        return FILE_TYPE_XML
    return FILE_TYPE_UNSUPPORTED  # 0条/无String判unsupported


def classify_translation_file(file_path: str) -> tuple[str, Optional[str]]:
    # 返回(类型,原因)，Unsafe原因透出便于展示
    from src.safe_xml import UnsafeXMLDeclarationError

    try:
        return detect_translation_file_type(file_path), None
    except UnsafeXMLDeclarationError as e:
        return FILE_TYPE_UNSUPPORTED, str(e)