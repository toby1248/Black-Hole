#!/usr/bin/env python3
"""
Preferences Dialog for TDE-SPH GUI (TASK4).

Modal dialog for configuring application settings that persist via QSettings.

Categories:
- General: Default directories, auto-save, confirmations
- Visualization: Colormap, point size, camera defaults
- Performance: GPU usage, thread count, memory limits

Author: TDE-SPH Development Team
Date: 2025-11-21
"""

import os
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
        QFormLayout, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QDialogButtonBox, QFileDialog,
        QLabel, QGroupBox, QMessageBox
    )
    from PyQt6.QtCore import QSettings
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
        QFormLayout, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QDialogButtonBox, QFileDialog,
        QLabel, QGroupBox, QMessageBox
    )
    from PyQt5.QtCore import QSettings
    PYQT_VERSION = 5


# Default settings values
DEFAULTS = {
    # General
    'default_config_dir': os.getcwd(),
    'auto_save_interval': 5,  # minutes
    'show_confirmations': True,
    'recent_files_count': 10,

    # Visualization
    'default_colormap': 'viridis',
    'default_point_size': 2.0,
    'camera_x': 50.0,
    'camera_y': 50.0,
    'camera_z': 50.0,
    'show_axes': True,
    'show_black_hole': True,

    # Performance
    'prefer_gpu': True,
    'thread_count': 0,  # 0 = auto
    'progress_update_steps': 100,
    'memory_limit_mb': 0,  # 0 = unlimited
}


class PreferencesDialog(QDialog):
    """
    Preferences dialog with tabbed settings categories.

    Categories:
    - General: Application-wide behavior
    - Visualization: Default rendering settings
    - Performance: Resource usage preferences

    Uses QSettings for persistence across sessions.
    """

    def __init__(self, parent=None):
        """Initialize preferences dialog."""
        super().__init__(parent)

        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)

        # Main layout
        layout = QVBoxLayout()

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self._create_general_tab()
        self._create_visualization_tab()
        self._create_performance_tab()

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)

        # Get Apply button and connect
        apply_button = button_box.button(QDialogButtonBox.StandardButton.Apply)
        if apply_button:
            apply_button.clicked.connect(self._apply_settings)

        layout.addWidget(button_box)
        self.setLayout(layout)

        # Load current settings
        self._load_settings()

    def _create_general_tab(self):
        """Create General settings tab."""
        tab = QWidget()
        layout = QFormLayout()

        # Default config directory
        dir_layout = QHBoxLayout()
        self.config_dir_edit = QLineEdit()
        self.config_dir_edit.setPlaceholderText("Select default configuration directory")
        dir_layout.addWidget(self.config_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_config_dir)
        dir_layout.addWidget(browse_btn)

        dir_widget = QWidget()
        dir_widget.setLayout(dir_layout)
        layout.addRow("Default Config Directory:", dir_widget)

        # Auto-save interval
        self.auto_save_spin = QSpinBox()
        self.auto_save_spin.setRange(0, 60)
        self.auto_save_spin.setSuffix(" min")
        self.auto_save_spin.setSpecialValueText("Disabled")
        self.auto_save_spin.setToolTip("Auto-save interval (0 = disabled)")
        layout.addRow("Auto-save Interval:", self.auto_save_spin)

        # Show confirmations
        self.show_confirmations_check = QCheckBox("Show confirmation dialogs")
        self.show_confirmations_check.setToolTip("Show confirmation dialogs before destructive actions")
        layout.addRow("", self.show_confirmations_check)

        # Recent files count
        self.recent_files_spin = QSpinBox()
        self.recent_files_spin.setRange(0, 50)
        self.recent_files_spin.setToolTip("Number of recent files to remember (0 = disabled)")
        layout.addRow("Recent Files Count:", self.recent_files_spin)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "General")

    def _create_visualization_tab(self):
        """Create Visualization settings tab."""
        tab = QWidget()
        layout = QFormLayout()

        # Default colormap
        self.colormap_combo = QComboBox()
        colormaps = ['viridis', 'plasma', 'inferno', 'hot', 'cool', 'jet', 'rainbow']
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setToolTip("Default colormap for particle visualization")
        layout.addRow("Default Colormap:", self.colormap_combo)

        # Default point size
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.5, 10.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setToolTip("Default particle point size in visualization")
        layout.addRow("Default Point Size:", self.point_size_spin)

        # Camera position group
        camera_group = QGroupBox("Default Camera Position")
        camera_layout = QFormLayout()

        self.camera_x_spin = QDoubleSpinBox()
        self.camera_x_spin.setRange(-1000, 1000)
        self.camera_x_spin.setDecimals(1)
        camera_layout.addRow("X:", self.camera_x_spin)

        self.camera_y_spin = QDoubleSpinBox()
        self.camera_y_spin.setRange(-1000, 1000)
        self.camera_y_spin.setDecimals(1)
        camera_layout.addRow("Y:", self.camera_y_spin)

        self.camera_z_spin = QDoubleSpinBox()
        self.camera_z_spin.setRange(-1000, 1000)
        self.camera_z_spin.setDecimals(1)
        camera_layout.addRow("Z:", self.camera_z_spin)

        camera_group.setLayout(camera_layout)
        layout.addRow(camera_group)

        # Show options
        self.show_axes_check = QCheckBox("Show coordinate axes by default")
        layout.addRow("", self.show_axes_check)

        self.show_bh_check = QCheckBox("Show black hole marker by default")
        layout.addRow("", self.show_bh_check)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Visualization")

    def _create_performance_tab(self):
        """Create Performance settings tab."""
        tab = QWidget()
        layout = QFormLayout()

        # GPU preference
        self.gpu_check = QCheckBox("Prefer GPU (CUDA) if available")
        self.gpu_check.setToolTip("Use GPU acceleration when available")
        layout.addRow("", self.gpu_check)

        # Thread count
        self.thread_count_spin = QSpinBox()
        max_threads = os.cpu_count() or 8
        self.thread_count_spin.setRange(0, max_threads * 2)
        self.thread_count_spin.setSpecialValueText("Auto")
        self.thread_count_spin.setToolTip(f"Number of threads (0 = auto, max recommended: {max_threads})")
        layout.addRow("Thread Count:", self.thread_count_spin)

        # Progress update frequency
        self.progress_steps_spin = QSpinBox()
        self.progress_steps_spin.setRange(1, 10000)
        self.progress_steps_spin.setToolTip("Number of simulation steps between progress updates")
        layout.addRow("Progress Update Steps:", self.progress_steps_spin)

        # Memory limit
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(0, 65536)
        self.memory_limit_spin.setSuffix(" MB")
        self.memory_limit_spin.setSpecialValueText("Unlimited")
        self.memory_limit_spin.setToolTip("Memory limit for simulation (0 = unlimited)")
        layout.addRow("Memory Limit:", self.memory_limit_spin)

        # Info label
        info_label = QLabel(
            f"<i>System: {os.cpu_count() or 'Unknown'} CPU cores</i>"
        )
        layout.addRow("", info_label)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Performance")

    def _browse_config_dir(self):
        """Open directory browser for config directory."""
        current = self.config_dir_edit.text() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Default Configuration Directory",
            current
        )
        if directory:
            self.config_dir_edit.setText(directory)

    def _load_settings(self):
        """Load settings from QSettings."""
        settings = QSettings('TDE-SPH', 'Simulator')

        # General
        self.config_dir_edit.setText(
            settings.value('default_config_dir', DEFAULTS['default_config_dir'])
        )
        self.auto_save_spin.setValue(
            int(settings.value('auto_save_interval', DEFAULTS['auto_save_interval']))
        )
        self.show_confirmations_check.setChecked(
            settings.value('show_confirmations', DEFAULTS['show_confirmations']) in (True, 'true', '1')
        )
        self.recent_files_spin.setValue(
            int(settings.value('recent_files_count', DEFAULTS['recent_files_count']))
        )

        # Visualization
        colormap = settings.value('default_colormap', DEFAULTS['default_colormap'])
        index = self.colormap_combo.findText(colormap)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)

        self.point_size_spin.setValue(
            float(settings.value('default_point_size', DEFAULTS['default_point_size']))
        )
        self.camera_x_spin.setValue(
            float(settings.value('camera_x', DEFAULTS['camera_x']))
        )
        self.camera_y_spin.setValue(
            float(settings.value('camera_y', DEFAULTS['camera_y']))
        )
        self.camera_z_spin.setValue(
            float(settings.value('camera_z', DEFAULTS['camera_z']))
        )
        self.show_axes_check.setChecked(
            settings.value('show_axes', DEFAULTS['show_axes']) in (True, 'true', '1')
        )
        self.show_bh_check.setChecked(
            settings.value('show_black_hole', DEFAULTS['show_black_hole']) in (True, 'true', '1')
        )

        # Performance
        self.gpu_check.setChecked(
            settings.value('prefer_gpu', DEFAULTS['prefer_gpu']) in (True, 'true', '1')
        )
        self.thread_count_spin.setValue(
            int(settings.value('thread_count', DEFAULTS['thread_count']))
        )
        self.progress_steps_spin.setValue(
            int(settings.value('progress_update_steps', DEFAULTS['progress_update_steps']))
        )
        self.memory_limit_spin.setValue(
            int(settings.value('memory_limit_mb', DEFAULTS['memory_limit_mb']))
        )

    def _apply_settings(self) -> bool:
        """
        Save settings to QSettings.

        Returns:
            True if settings saved successfully, False if validation failed
        """
        # Validate
        if not self._validate_settings():
            return False

        settings = QSettings('TDE-SPH', 'Simulator')

        # General
        settings.setValue('default_config_dir', self.config_dir_edit.text())
        settings.setValue('auto_save_interval', self.auto_save_spin.value())
        settings.setValue('show_confirmations', self.show_confirmations_check.isChecked())
        settings.setValue('recent_files_count', self.recent_files_spin.value())

        # Visualization
        settings.setValue('default_colormap', self.colormap_combo.currentText())
        settings.setValue('default_point_size', self.point_size_spin.value())
        settings.setValue('camera_x', self.camera_x_spin.value())
        settings.setValue('camera_y', self.camera_y_spin.value())
        settings.setValue('camera_z', self.camera_z_spin.value())
        settings.setValue('show_axes', self.show_axes_check.isChecked())
        settings.setValue('show_black_hole', self.show_bh_check.isChecked())

        # Performance
        settings.setValue('prefer_gpu', self.gpu_check.isChecked())
        settings.setValue('thread_count', self.thread_count_spin.value())
        settings.setValue('progress_update_steps', self.progress_steps_spin.value())
        settings.setValue('memory_limit_mb', self.memory_limit_spin.value())

        settings.sync()
        return True

    def _validate_settings(self) -> bool:
        """
        Validate settings before saving.

        Returns:
            True if all settings are valid
        """
        # Validate config directory exists (if specified)
        config_dir = self.config_dir_edit.text()
        if config_dir and not os.path.isdir(config_dir):
            QMessageBox.warning(
                self,
                "Invalid Directory",
                f"The specified config directory does not exist:\n{config_dir}"
            )
            return False

        return True

    def _on_ok(self):
        """Handle OK button click."""
        if self._apply_settings():
            self.accept()


def get_preference(key: str, default=None):
    """
    Get a preference value from QSettings.

    Parameters:
        key: Setting key name
        default: Default value if not set

    Returns:
        Setting value
    """
    settings = QSettings('TDE-SPH', 'Simulator')
    return settings.value(key, default if default is not None else DEFAULTS.get(key))


def set_preference(key: str, value):
    """
    Set a preference value in QSettings.

    Parameters:
        key: Setting key name
        value: Value to set
    """
    settings = QSettings('TDE-SPH', 'Simulator')
    settings.setValue(key, value)
    settings.sync()
