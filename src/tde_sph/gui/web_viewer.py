#!/usr/bin/env python3
"""
Web Viewer Widget for TDE-SPH GUI (TASK5).

Embeds the web-based 3D visualization using QWebEngineView.
Allows loading simulation snapshots and provides refresh functionality.

Author: TDE-SPH Development Team
Date: 2025-11-21
"""

import os
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QLabel, QFileDialog, QMessageBox, QComboBox
    )
    from PyQt6.QtCore import QUrl, pyqtSignal
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEB_ENGINE = True
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
            QLabel, QFileDialog, QMessageBox, QComboBox
        )
        from PyQt5.QtCore import QUrl, pyqtSignal
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        HAS_WEB_ENGINE = True
        PYQT_VERSION = 5
    except ImportError:
        # QWebEngineView not available
        HAS_WEB_ENGINE = False
        QWidget = object  # Placeholder
        pyqtSignal = lambda *args: None


class WebViewerWidget(QWidget):
    """
    Widget that embeds web-based 3D visualization.

    Uses QWebEngineView to display the Plotly-based web visualization.
    Can load simulation snapshots from HDF5 files.

    Signals:
        snapshot_loaded (str): Emitted when a snapshot is loaded (path)
        error_occurred (str): Emitted on error

    Note:
        Requires PyQt6-WebEngine or PyQt5-WebEngine package.
    """

    snapshot_loaded = pyqtSignal(str) if HAS_WEB_ENGINE else None
    error_occurred = pyqtSignal(str) if HAS_WEB_ENGINE else None

    def __init__(self, parent=None):
        """Initialize web viewer widget."""
        if not HAS_WEB_ENGINE:
            raise ImportError(
                "QWebEngineView not available. Install with:\n"
                "  pip install PyQt6-WebEngine  # For PyQt6\n"
                "  pip install PyQtWebEngine     # For PyQt5"
            )

        super().__init__(parent)

        self.current_snapshot: Optional[Path] = None
        self._web_dir = self._find_web_directory()

        self._setup_ui()
        self._load_viewer()

    def _find_web_directory(self) -> Optional[Path]:
        """Find the web/ directory containing index.html."""
        # Try relative to this file
        module_dir = Path(__file__).parent
        possible_paths = [
            module_dir.parent.parent.parent / 'web',  # src/tde_sph/gui -> web
            module_dir.parent.parent / 'web',
            Path.cwd() / 'web',
        ]

        for path in possible_paths:
            index_file = path / 'index.html'
            if index_file.exists():
                return path

        return None

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Toolbar
        toolbar = QHBoxLayout()

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload the current view")
        refresh_btn.clicked.connect(self._refresh)
        toolbar.addWidget(refresh_btn)

        # Load snapshot button
        load_btn = QPushButton("Load Snapshot...")
        load_btn.setToolTip("Load simulation snapshot (HDF5)")
        load_btn.clicked.connect(self._load_snapshot)
        toolbar.addWidget(load_btn)

        # Colormap selector
        toolbar.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'hot', 'cool'])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        toolbar.addWidget(self.colormap_combo)

        toolbar.addStretch()

        # Status label
        self.status_label = QLabel("")
        toolbar.addWidget(self.status_label)

        layout.addLayout(toolbar)

        # Web view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(800, 600)
        layout.addWidget(self.web_view)

        self.setLayout(layout)

    def _load_viewer(self):
        """Load the web viewer HTML."""
        if self._web_dir is None:
            self._show_error_page("Web directory not found")
            return

        index_file = self._web_dir / 'index.html'
        if not index_file.exists():
            self._show_error_page(f"index.html not found in {self._web_dir}")
            return

        # Load the HTML file
        url = QUrl.fromLocalFile(str(index_file.absolute()))
        self.web_view.setUrl(url)
        self.status_label.setText("Viewer loaded")

    def _show_error_page(self, message: str):
        """Show error message in web view."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: #f0f0f0;
                }}
                .error {{
                    text-align: center;
                    padding: 40px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h2 {{ color: #c00; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>Web Viewer Error</h2>
                <p>{message}</p>
                <p><small>Please ensure the web/ directory is available.</small></p>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(html)
        self.status_label.setText("Error")
        if self.error_occurred:
            self.error_occurred.emit(message)

    def _refresh(self):
        """Refresh the web view."""
        self.web_view.reload()
        self.status_label.setText("Refreshed")

    def _load_snapshot(self):
        """Open file dialog to load a snapshot."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Simulation Snapshot",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )

        if file_path:
            self.load_snapshot_file(file_path)

    def load_snapshot_file(self, file_path: str):
        """
        Load a snapshot file into the viewer.

        Parameters:
            file_path: Path to HDF5 snapshot file
        """
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"Snapshot file not found:\n{file_path}")
            return

        self.current_snapshot = path
        self.status_label.setText(f"Loaded: {path.name}")

        # Pass snapshot path to web app via JavaScript
        # The web app should listen for this
        js_code = f'if(typeof loadSnapshot === "function") {{ loadSnapshot("{path.absolute()}"); }}'
        self.web_view.page().runJavaScript(js_code)

        if self.snapshot_loaded:
            self.snapshot_loaded.emit(str(path))

    def _on_colormap_changed(self, colormap: str):
        """Handle colormap selection change."""
        # Pass colormap to web app
        js_code = f'if(typeof setColormap === "function") {{ setColormap("{colormap}"); }}'
        self.web_view.page().runJavaScript(js_code)

    def set_output_directory(self, output_dir: str):
        """
        Set the output directory for snapshots.

        Parameters:
            output_dir: Path to simulation output directory
        """
        js_code = f'if(typeof setOutputDir === "function") {{ setOutputDir("{output_dir}"); }}'
        self.web_view.page().runJavaScript(js_code)


class WebViewerPlaceholder(QWidget):
    """
    Placeholder widget when QWebEngineView is not available.

    Shows a message explaining how to install the required package.
    """

    def __init__(self, parent=None):
        """Initialize placeholder widget."""
        # Need to import QWidget differently if HAS_WEB_ENGINE is False
        try:
            from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
        except ImportError:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

        super(QWidget, self).__init__(parent)

        layout = QVBoxLayout()

        message = QLabel(
            "<h3>Web Viewer Not Available</h3>"
            "<p>The web-based 3D viewer requires PyQt WebEngine.</p>"
            "<p>Install with:</p>"
            "<pre>pip install PyQt6-WebEngine  # For PyQt6</pre>"
            "<pre>pip install PyQtWebEngine     # For PyQt5</pre>"
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        layout.addStretch()
        self.setLayout(layout)


def create_web_viewer(parent=None):
    """
    Factory function to create web viewer widget.

    Returns WebViewerWidget if available, otherwise WebViewerPlaceholder.

    Parameters:
        parent: Parent widget

    Returns:
        WebViewerWidget or WebViewerPlaceholder
    """
    if HAS_WEB_ENGINE:
        try:
            return WebViewerWidget(parent)
        except Exception as e:
            print(f"Failed to create WebViewerWidget: {e}")
            return WebViewerPlaceholder(parent)
    else:
        return WebViewerPlaceholder(parent)
