#!/usr/bin/env python3
"""
YAML Configuration Editor Widget (TASK-100)

A text editor widget with YAML syntax highlighting, line numbers,
and validation feedback for editing TDE-SPH configuration files.

Features:
- Syntax highlighting for YAML (keys, values, comments)
- Line number display
- Undo/redo support
- Find and replace
- Automatic indentation
- Validation indicators

Author: TDE-SPH Development Team
Date: 2025-11-18
"""

from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QPlainTextEdit, QTextEdit, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit
    )
    from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal
    from PyQt6.QtGui import (
        QColor, QPainter, QTextFormat, QFont, QSyntaxHighlighter,
        QTextCharFormat, QPalette, QTextCursor
    )
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QPlainTextEdit, QTextEdit, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit
    )
    from PyQt5.QtCore import Qt, QRect, QSize, pyqtSignal
    from PyQt5.QtGui import (
        QColor, QPainter, QTextFormat, QFont, QSyntaxHighlighter,
        QTextCharFormat, QPalette, QTextCursor
    )
    PYQT_VERSION = 5

import re


class YAMLSyntaxHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter for YAML files.

    Highlights:
    - Keys (before colon) in blue
    - String values in green
    - Numbers in orange
    - Comments in gray
    - Special keywords (true, false, null) in purple
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Define highlighting formats
        self.formats = {}

        # Keys (before colon)
        key_format = QTextCharFormat()
        key_format.setForeground(QColor(0, 102, 204))  # Blue
        key_format.setFontWeight(QFont.Weight.Bold)
        self.formats['key'] = key_format

        # String values
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(0, 153, 0))  # Green
        self.formats['string'] = string_format

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(255, 102, 0))  # Orange
        self.formats['number'] = number_format

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(128, 128, 128))  # Gray
        comment_format.setFontItalic(True)
        self.formats['comment'] = comment_format

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(153, 0, 153))  # Purple
        keyword_format.setFontWeight(QFont.Weight.Bold)
        self.formats['keyword'] = keyword_format

        # List markers
        list_format = QTextCharFormat()
        list_format.setForeground(QColor(204, 0, 0))  # Red
        self.formats['list'] = list_format

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a single line of text."""
        # Comments (highest priority, overrides everything)
        comment_match = re.search(r'#.*$', text)
        if comment_match:
            self.setFormat(
                comment_match.start(),
                len(comment_match.group()),
                self.formats['comment']
            )
            # Don't highlight the rest of the line
            text = text[:comment_match.start()]

        # Keys (word before colon)
        for match in re.finditer(r'^(\s*)([a-zA-Z_][\w]*)\s*:', text):
            key_start = match.start(2)
            key_length = len(match.group(2))
            self.setFormat(key_start, key_length, self.formats['key'])

        # List markers
        for match in re.finditer(r'^\s*-\s', text):
            self.setFormat(match.start(), match.end() - match.start(), self.formats['list'])

        # Keywords (true, false, null, yes, no)
        keywords = ['true', 'false', 'null', 'yes', 'no', 'True', 'False', 'None']
        for keyword in keywords:
            for match in re.finditer(r'\b' + keyword + r'\b', text):
                self.setFormat(match.start(), len(keyword), self.formats['keyword'])

        # Numbers (integers and floats, including scientific notation)
        number_pattern = r'\b-?\d+\.?\d*([eE][+-]?\d+)?\b'
        for match in re.finditer(number_pattern, text):
            self.setFormat(match.start(), len(match.group()), self.formats['number'])

        # Strings (quoted strings)
        # Double-quoted strings
        for match in re.finditer(r'"[^"]*"', text):
            self.setFormat(match.start(), len(match.group()), self.formats['string'])

        # Single-quoted strings
        for match in re.finditer(r"'[^']*'", text):
            self.setFormat(match.start(), len(match.group()), self.formats['string'])


class LineNumberArea(QWidget):
    """Widget for displaying line numbers alongside the text editor."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class YAMLTextEdit(QPlainTextEdit):
    """
    Enhanced text editor for YAML files with line numbers and auto-indentation.
    """

    config_modified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Font
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.setFont(font)

        # Tab settings
        self.setTabStopDistance(40)  # 4 spaces

        # Line wrapping
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Line number area
        self.line_number_area = LineNumberArea(self)

        # Connect signals
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.textChanged.connect(self.on_text_changed)

        # Initialize
        self.update_line_number_area_width(0)
        self.highlight_current_line()

        # Syntax highlighter
        self.highlighter = YAMLSyntaxHighlighter(self.document())

        # Modification tracking
        self.is_modified_flag = False

    def line_number_area_width(self):
        """Calculate width needed for line numbers."""
        digits = len(str(max(1, self.blockCount())))
        space = 3 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def update_line_number_area_width(self, _):
        """Update margins to accommodate line numbers."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        """Update line number area when scrolling."""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)

        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        """Paint line numbers."""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(240, 240, 240))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor(100, 100, 100))
                painter.drawText(
                    0, top,
                    self.line_number_area.width() - 3,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number
                )

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self):
        """Highlight the line where the cursor is."""
        extra_selections = []

        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()

            line_color = QColor(255, 255, 220)  # Light yellow
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()

            extra_selections.append(selection)

        self.setExtraSelections(extra_selections)

    def keyPressEvent(self, event):
        """Handle key press events for auto-indentation."""
        # Auto-indent on Enter
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            cursor = self.textCursor()
            block = cursor.block()
            text = block.text()

            # Calculate indentation of current line
            indent = len(text) - len(text.lstrip())

            # If line ends with colon, increase indent
            if text.rstrip().endswith(':'):
                indent += 2

            super().keyPressEvent(event)

            # Insert indentation
            self.insertPlainText(' ' * indent)
            return

        # Tab to spaces
        elif event.key() == Qt.Key.Key_Tab:
            self.insertPlainText('  ')  # 2 spaces
            return

        super().keyPressEvent(event)

    def on_text_changed(self):
        """Handle text changes."""
        self.is_modified_flag = True
        self.config_modified.emit()

    def is_modified(self):
        """Check if text has been modified."""
        return self.is_modified_flag

    def reset_modified_flag(self):
        """Reset the modification flag (after saving)."""
        self.is_modified_flag = False


class ConfigEditorWidget(QWidget):
    """
    Configuration editor widget with YAML syntax highlighting and validation.

    Provides:
    - Text editor with line numbers
    - Syntax highlighting
    - Validation button
    - Undo/redo
    """

    config_modified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout()

        # Text editor
        self.text_edit = YAMLTextEdit()
        self.text_edit.config_modified.connect(self.config_modified.emit)
        layout.addWidget(self.text_edit)

        # Status bar (validation feedback)
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { padding: 5px; }")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        validate_button = QPushButton("Validate YAML")
        validate_button.clicked.connect(self.validate_yaml)
        status_layout.addWidget(validate_button)

        layout.addLayout(status_layout)

        self.setLayout(layout)

    def set_text(self, text: str):
        """Set editor text."""
        self.text_edit.setPlainText(text)
        self.text_edit.reset_modified_flag()

    def get_text(self) -> str:
        """Get editor text."""
        return self.text_edit.toPlainText()

    def is_modified(self) -> bool:
        """Check if text has been modified."""
        return self.text_edit.is_modified()

    def undo(self):
        """Undo last edit."""
        self.text_edit.undo()

    def redo(self):
        """Redo last undone edit."""
        self.text_edit.redo()

    def validate_yaml(self):
        """Validate YAML syntax."""
        try:
            import yaml

            content = self.get_text()
            yaml.safe_load(content)

            self.status_label.setText("✓ YAML is valid")
            self.status_label.setStyleSheet("QLabel { color: green; padding: 5px; font-weight: bold; }")

        except yaml.YAMLError as e:
            self.status_label.setText(f"✗ YAML Error: {str(e)[:50]}...")
            self.status_label.setStyleSheet("QLabel { color: red; padding: 5px; font-weight: bold; }")

        except Exception as e:
            self.status_label.setText(f"✗ Error: {str(e)[:50]}...")
            self.status_label.setStyleSheet("QLabel { color: red; padding: 5px; font-weight: bold; }")
