import json
import os
import locale
import sys
import threading

_warned_missing: set[str] = set()


class I18n:
    def __init__(self):
        self._lock = threading.RLock()
        self.translations: dict = {}
        self._en_translations: dict = {}
        self.current_lang = 'en'
        if getattr(sys, 'frozen', False):
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.locale_dir = os.path.join(base_path, 'locales')
        self._en_translations = self._load_file('en')
        self.load_language()
        self._validate_key_parity()

    def _load_file(self, lang_code: str) -> dict:
        file_path = os.path.join(self.locale_dir, f'{lang_code}.json')
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"Error loading language file {file_path}: {e}")
            return {}

    def _detect_system_lang(self) -> str:
        try:
            sys_lang = (locale.getlocale()[0] or "")
        except Exception:
            sys_lang = ""
        if not sys_lang:
            try:
                sys_lang = (locale.getdefaultlocale()[0] or "")
            except Exception:
                sys_lang = ""
        low = sys_lang.lower().replace("-", "_")
        if low.startswith("zh"):
            return 'zh'
        return 'en'

    def _validate_key_parity(self) -> None:
        try:
            with self._lock:
                zh = self._load_file('zh')
                en_keys = set(self._en_translations or {})
                zh_keys = set(zh or {})
                only_en = sorted(en_keys - zh_keys)
                only_zh = sorted(zh_keys - en_keys)
            if only_en:
                print(f"[i18n] WARNING keys only in en: {only_en[:20]}")
            if only_zh:
                print(f"[i18n] WARNING keys only in zh: {only_zh[:20]}")
        except Exception:
            pass

    def load_language(self, lang_code=None):
        with self._lock:
            if not lang_code or lang_code == 'auto':
                lang_code = self._detect_system_lang()
            norm = str(lang_code).replace("-", "_")
            if norm.lower() in ("zh_hant", "zh_tw", "zh_hk"):
                lang_code = 'zh'
            data = self._load_file(lang_code)
            if data:
                self.translations = data
                self.current_lang = lang_code
                return
            if lang_code != 'en':
                print(f"[i18n] WARNING locale '{lang_code}' missing/corrupted, falling back to en")
                self.translations = dict(self._en_translations)
                self.current_lang = 'en'
            else:
                self.translations = dict(self._en_translations)
                self.current_lang = 'en'

    def t(self, key, default=None, **kwargs):
        with self._lock:
            if key in self.translations:
                text = self.translations[key]
            elif key in self._en_translations:
                text = self._en_translations[key]
            else:
                text = default if default is not None else key
                if key not in _warned_missing:
                    _warned_missing.add(key)
                    print(f"[i18n] WARNING missing key '{key}'")
                return self.safe_format(text, **kwargs) if kwargs else text
            if key not in self.translations and key in self._en_translations:
                if key not in _warned_missing:
                    _warned_missing.add(key)
                    print(f"[i18n] WARNING key '{key}' missing in '{self.current_lang}', using en")
        return self.safe_format(text, **kwargs) if kwargs else text

    @staticmethod
    def safe_format(text, **kwargs) -> str:
        if not isinstance(text, str) or not kwargs:
            return text
        try:
            return text.format(**kwargs)
        except Exception as e:
            print(f"[i18n] WARNING format failed for {text!r}: {e}")
            return text

# Global instance
i18n = I18n()
