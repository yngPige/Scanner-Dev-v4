"""
Entry point for the 3lacks Trading Terminal PyQt6 GUI.
"""
import sys
from PyQt6.QtWidgets import QApplication
from Features.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 