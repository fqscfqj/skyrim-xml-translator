import os


FILE_TYPE_XML = "xml"
FILE_TYPE_ESP_XML = "esp_xml"
FILE_TYPE_MCM = "mcm"
FILE_TYPE_RAW_PLUGIN = "raw_plugin"
FILE_TYPE_UNSUPPORTED = "unsupported"

RAW_PLUGIN_EXTENSIONS = frozenset({".esp", ".esm", ".esl"})


def normalize_extension(file_path: str) -> str:
    return os.path.splitext(str(file_path or ""))[1].lower()


def describe_extension(file_path: str) -> str:
    ext = normalize_extension(file_path)
    return ext or "(no extension)"


def detect_translation_file_type_from_extension(file_path: str) -> str:
    ext = normalize_extension(file_path)
    if ext == ".txt":
        return FILE_TYPE_MCM
    if ext == ".xml":
        return FILE_TYPE_XML
    if ext in RAW_PLUGIN_EXTENSIONS:
        return FILE_TYPE_RAW_PLUGIN
    return FILE_TYPE_UNSUPPORTED