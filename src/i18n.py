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
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.locale_dir = os.path.join(base_path, 'locales')
        self.load_language()

    def load_language(self, lang_code=None):
        if lang_code is None:
            # Detect system language
            sys_lang = locale.getdefaultlocale()[0]
            if sys_lang and sys_lang.startswith('zh'):
                lang_code = 'zh'
            else:
                lang_code = 'en'
        
        self.current_lang = lang_code
        file_path = os.path.join(self.locale_dir, f'{lang_code}.json')
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
            except Exception as e:
                print(f"Error loading language file {file_path}: {e}")
                self.translations = {}
        else:
            # Fallback to empty if file not found, effectively using keys as default (or English if keys are English)
            self.translations = {}

    def t(self, key, default=None):
        return self.translations.get(key, default if default is not None else key)

# Global instance
i18n = I18n()
