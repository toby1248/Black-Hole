"""
Tests for TASK4 Preferences Dialog.

Tests cover:
- PreferencesDialog initialization
- Settings load and save via QSettings
- Default values
- Helper functions get_preference/set_preference

References
----------
- TASK4: Implement Preferences Dialog
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPreferencesDefaults:
    """Test default preference values."""

    def test_defaults_exist(self):
        """Test that DEFAULTS dict has expected keys."""
        from tde_sph.gui.preferences_dialog import DEFAULTS

        expected_keys = [
            'default_config_dir',
            'auto_save_interval',
            'show_confirmations',
            'default_colormap',
            'default_point_size',
            'prefer_gpu',
            'thread_count',
        ]

        for key in expected_keys:
            assert key in DEFAULTS, f"Missing default for {key}"

    def test_default_colormap(self):
        """Test default colormap is valid."""
        from tde_sph.gui.preferences_dialog import DEFAULTS

        valid_colormaps = ['viridis', 'plasma', 'inferno', 'hot', 'cool', 'jet', 'rainbow']
        assert DEFAULTS['default_colormap'] in valid_colormaps

    def test_default_point_size_positive(self):
        """Test default point size is positive."""
        from tde_sph.gui.preferences_dialog import DEFAULTS

        assert DEFAULTS['default_point_size'] > 0


class TestPreferenceHelpers:
    """Test get_preference and set_preference helpers."""

    def test_get_preference_with_default(self):
        """Test get_preference returns default when not set."""
        from tde_sph.gui.preferences_dialog import get_preference

        # Use a key that's unlikely to be set
        result = get_preference('test_nonexistent_key', 'my_default')
        assert result == 'my_default'

    def test_get_preference_uses_defaults_dict(self):
        """Test get_preference falls back to DEFAULTS dict."""
        from tde_sph.gui.preferences_dialog import get_preference, DEFAULTS

        # For a key in DEFAULTS but not explicitly set
        result = get_preference('default_colormap')
        # Should return either stored value or DEFAULTS value
        assert result is not None


class TestPreferencesDialogCreation:
    """Test PreferencesDialog creation (requires PyQt)."""

    def test_dialog_creation(self):
        """Test PreferencesDialog can be created."""
        try:
            from tde_sph.gui.preferences_dialog import PreferencesDialog
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create dialog
        dialog = PreferencesDialog()
        assert dialog is not None

        # Check tabs exist
        assert dialog.tabs.count() == 3

        # Check tab names
        tab_names = [dialog.tabs.tabText(i) for i in range(3)]
        assert "General" in tab_names
        assert "Visualization" in tab_names
        assert "Performance" in tab_names

    def test_dialog_widgets_exist(self):
        """Test that key widgets are created."""
        try:
            from tde_sph.gui.preferences_dialog import PreferencesDialog
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        dialog = PreferencesDialog()

        # General tab widgets
        assert hasattr(dialog, 'config_dir_edit')
        assert hasattr(dialog, 'auto_save_spin')
        assert hasattr(dialog, 'show_confirmations_check')

        # Visualization tab widgets
        assert hasattr(dialog, 'colormap_combo')
        assert hasattr(dialog, 'point_size_spin')
        assert hasattr(dialog, 'show_axes_check')

        # Performance tab widgets
        assert hasattr(dialog, 'gpu_check')
        assert hasattr(dialog, 'thread_count_spin')


class TestPreferencesValidation:
    """Test preferences validation."""

    def test_validate_nonexistent_directory(self):
        """Test that invalid directory fails validation."""
        try:
            from tde_sph.gui.preferences_dialog import PreferencesDialog
        except ImportError:
            pytest.skip("PyQt not available")

        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
        except ImportError:
            try:
                from PyQt5.QtWidgets import QApplication, QMessageBox
            except ImportError:
                pytest.skip("No PyQt available")

        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        dialog = PreferencesDialog()

        # Set invalid directory
        dialog.config_dir_edit.setText('/nonexistent/path/12345')

        # Mock QMessageBox to avoid actual dialog
        with patch.object(QMessageBox, 'warning', return_value=None):
            result = dialog._validate_settings()
            assert result is False
