import sys
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.gui_main import MainWindow
from src.logging_helper import emit as log_emit
from src.config_manager import ConfigManager

def main():
    # Setup a global exception handler to catch uncaught exceptions and log them.
    def excepthook(exc_type, exc_value, exc_traceback):
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        message = f"Uncaught exception: {exc_value}"
        log_emit(None, cfg, 'ERROR', message, exc=exc_value, module='main', func='excepthook')
        try:
            QMessageBox.critical(None, 'Unhandled Exception', message)
        except Exception:
            # If UI can't show, print to console as last resort
            print(message)
        # Call the default excepthook to ensure standard behavior
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
