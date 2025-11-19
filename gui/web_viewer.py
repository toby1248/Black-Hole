
"""
Web-based 3D Viewer Widget for TDE-SPH GUI.

Embeds the Three.js visualization (from web/index.html) into the PyQt application.
"""

from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QLabel
from PyQt6.QtCore import QUrl

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

class WebViewerWidget(QWidget):
    """
    Widget that embeds a web browser to show the Three.js visualization.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        if not HAS_WEBENGINE:
            self.layout.addWidget(QLabel("Error: PyQt6-WebEngine is not installed.\n"
                                       "Please install it with: pip install PyQt6-WebEngine"))
            return
            
        self.browser = QWebEngineView()
        self.layout.addWidget(self.browser)
        
        # Load the local index.html
        # Assuming web/ is at the project root
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent
        web_dir = root_dir / "web"
        index_file = web_dir / "index.html"
        
        if index_file.exists():
            self.browser.load(QUrl.fromLocalFile(str(index_file)))
        else:
            self.layout.addWidget(QLabel(f"Error: Could not find {index_file}"))

    def reload(self):
        if HAS_WEBENGINE:
            self.browser.reload()

