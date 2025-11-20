import sys
import os
from PyQt6.QtWidgets import QApplication
from src.gui_main import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
