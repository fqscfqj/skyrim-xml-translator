import json
import os
import locale
import sys

class I18n:
    def __init__(self):
        self.translations = {}
        self.current_lang = 'en'
        
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            # Use getattr to avoid direct access to a private attribute (sys._MEIPASS),
            # which Pylance flags as an unknown attribute on sys.
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            # Running as script
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.locale_dir = os.path.join(base_path, 'locales')
        self.load_language()

    def load_language(self, lang_code=None):
        # Treat None or explicit "auto" as system language detection
        if not lang_code or lang_code == 'auto':
            sys_lang = locale.getdefaultlocale()[0]
            if sys_lang and sys_lang.startswith('zh'):
                lang_code = 'zh'
            else:
                lang_code = 'en'
        
        file_path = os.path.join(self.locale_dir, f'{lang_code}.json')
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
                self.current_lang = lang_code
            except Exception as e:
                print(f"Error loading language file {file_path}: {e}")
                self.translations = {}
                self.current_lang = lang_code
        else:
            # Fallback to English if requested language file is missing
            if lang_code != 'en':
                self.load_language('en')
            else:
                self.translations = {}
                self.current_lang = 'en'

    def t(self, key, default=None):
        return self.translations.get(key, default if default is not None else key)

# Global instance
i18n = I18n()
