# settings_dialogs.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Configuration and data-view settings dialogs
# ==============================================================================

from pathlib import Path
from typing import Dict, Optional

import yaml
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEPTH_COLORMAPS = (
    "grayscale",
    "jet",
    "turbo",
    "viridis",
    "inferno",
    "plasma",
    "magma",
)


def read_yaml_config(path: Path) -> Dict:
    """Read a YAML mapping and reject unsupported root values."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping.")
    return data


class DataViewSettingsDialog(QDialog):
    """Edit core data-view controls and preview rendering config."""

    def __init__(
        self,
        config_path: Path,
        config: Dict,
        view_count: int,
        plot_mode: str,
        zoom_seconds: float,
        parent=None,
    ):
        super().__init__(parent)
        self.config_path = config_path
        self.config = dict(config)
        self.action = ""
        self.setWindowTitle("Data View Settings")
        self.resize(520, 520)
        self._setup_ui()
        self._set_runtime_values(view_count, plot_mode, zoom_seconds)
        self._sync_depth_controls()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._config_path_widget())
        layout.addWidget(self._core_settings_group())
        layout.addWidget(self._depth_settings_group())
        layout.addStretch(1)
        layout.addWidget(self._button_box())

    def _config_path_widget(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit(str(self.config_path))
        self.path_edit.setReadOnly(True)
        self.path_edit.setCursorPosition(0)
        browse_button = QPushButton("Browse...")
        reload_button = QPushButton("Reload")
        browse_button.clicked.connect(self._browse_config)
        reload_button.clicked.connect(self._reload_config)
        layout.addWidget(QLabel("View Config"))
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(browse_button)
        layout.addWidget(reload_button)
        return container

    def _core_settings_group(self) -> QGroupBox:
        group = QGroupBox("Core Parameters")
        layout = QFormLayout(group)
        self.view_count_spin = QSpinBox()
        self.view_count_spin.setRange(1, 16)
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItem("Global cursor", "global_cursor")
        self.plot_mode_combo.addItem("Follow window", "follow_window")
        self.zoom_seconds_spin = QDoubleSpinBox()
        self.zoom_seconds_spin.setRange(0.1, 3600.0)
        self.zoom_seconds_spin.setDecimals(2)
        self.zoom_seconds_spin.setSuffix(" s")
        layout.addRow("Views", self.view_count_spin)
        layout.addRow("Plot Mode", self.plot_mode_combo)
        layout.addRow("Zoom Sec", self.zoom_seconds_spin)
        return group

    def _depth_settings_group(self) -> QGroupBox:
        group = QGroupBox("Depth Visualization")
        layout = QFormLayout(group)
        self.depth_enabled_check = QCheckBox("Colorize depth images")
        self.depth_colormap_combo = QComboBox()
        self.depth_colormap_combo.addItems(DEPTH_COLORMAPS)
        self.depth_range_combo = QComboBox()
        self.depth_range_combo.addItem("Manual", "manual")
        self.depth_range_combo.addItem("Automatic per frame", "auto")
        self.depth_min_spin = self._depth_value_spin()
        self.depth_max_spin = self._depth_value_spin()
        self.depth_contrast_spin = QDoubleSpinBox()
        self.depth_contrast_spin.setRange(0.05, 10.0)
        self.depth_contrast_spin.setDecimals(3)
        self.depth_contrast_spin.setSingleStep(0.05)
        self.depth_invalid_spin = self._depth_value_spin()
        layout.addRow(self.depth_enabled_check)
        layout.addRow("Colormap", self.depth_colormap_combo)
        layout.addRow("Range Mode", self.depth_range_combo)
        layout.addRow("Minimum", self.depth_min_spin)
        layout.addRow("Maximum", self.depth_max_spin)
        layout.addRow("Contrast Power", self.depth_contrast_spin)
        layout.addRow("Invalid Value", self.depth_invalid_spin)
        return group

    def _depth_value_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        spin.setDecimals(3)
        return spin

    def _button_box(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox()
        apply_button = buttons.addButton("Apply", QDialogButtonBox.ApplyRole)
        save_button = buttons.addButton("Save", QDialogButtonBox.AcceptRole)
        save_as_button = buttons.addButton("Save As...", QDialogButtonBox.ActionRole)
        cancel_button = buttons.addButton(QDialogButtonBox.Cancel)
        apply_button.clicked.connect(lambda: self._finish("apply"))
        save_button.clicked.connect(lambda: self._finish("save"))
        save_as_button.clicked.connect(self._save_as)
        cancel_button.clicked.connect(self.reject)
        return buttons

    def _set_runtime_values(
        self,
        view_count: int,
        plot_mode: str,
        zoom_seconds: float,
    ) -> None:
        self.view_count_spin.setValue(view_count)
        mode_index = self.plot_mode_combo.findData(plot_mode)
        self.plot_mode_combo.setCurrentIndex(max(mode_index, 0))
        self.zoom_seconds_spin.setValue(zoom_seconds)

    def _sync_depth_controls(self) -> None:
        depth = self.config.get("depth_visualization", {}) or {}
        self.depth_enabled_check.setChecked(bool(depth.get("enabled", True)))
        colormap = str(depth.get("colormap", "jet")).lower()
        colormap_index = self.depth_colormap_combo.findText(colormap)
        if colormap_index < 0:
            colormap_index = self.depth_colormap_combo.findText("jet")
        self.depth_colormap_combo.setCurrentIndex(colormap_index)
        range_index = self.depth_range_combo.findData(
            str(depth.get("range_mode", "auto")).lower()
        )
        self.depth_range_combo.setCurrentIndex(max(range_index, 0))
        self.depth_min_spin.setValue(float(depth.get("min_value", 0.0)))
        self.depth_max_spin.setValue(float(depth.get("max_value", 2000.0)))
        self.depth_contrast_spin.setValue(float(depth.get("contrast_power", 1.0)))
        self.depth_invalid_spin.setValue(float(depth.get("invalid_value", 0.0)))

    def result_config(self) -> Dict:
        """Merge dialog values into the selected view config."""
        data = dict(self.config)
        settings = dict(data.get("view_settings", {}) or {})
        settings.update({
            "view_count": self.view_count_spin.value(),
            "plot_mode": self.plot_mode_combo.currentData(),
            "zoom_seconds": self.zoom_seconds_spin.value(),
        })
        depth = dict(data.get("depth_visualization", {}) or {})
        depth.update({
            "enabled": self.depth_enabled_check.isChecked(),
            "colormap": self.depth_colormap_combo.currentText(),
            "range_mode": self.depth_range_combo.currentData(),
            "min_value": self.depth_min_spin.value(),
            "max_value": self.depth_max_spin.value(),
            "contrast_power": self.depth_contrast_spin.value(),
            "invalid_value": self.depth_invalid_spin.value(),
        })
        data["view_settings"] = settings
        data["depth_visualization"] = depth
        return data

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select View Config",
            str(self.config_path.parent),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self._load_path(Path(path))

    def _reload_config(self) -> None:
        self._load_path(self.config_path)

    def _load_path(self, path: Path) -> None:
        try:
            self.config = read_yaml_config(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        self.config_path = path
        self.path_edit.setText(str(path))
        self.path_edit.setCursorPosition(0)
        settings = self.config.get("view_settings", {}) or {}
        self._set_runtime_values(
            int(settings.get("view_count", self.view_count_spin.value())),
            str(settings.get("plot_mode", self.plot_mode_combo.currentData())),
            float(settings.get("zoom_seconds", self.zoom_seconds_spin.value())),
        )
        self._sync_depth_controls()

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save View Config As",
            str(self.config_path),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        self.config_path = self._yaml_path(Path(path))
        self.path_edit.setText(str(self.config_path))
        self.path_edit.setCursorPosition(0)
        self._finish("save")

    def _finish(self, action: str) -> None:
        if self.depth_max_spin.value() <= self.depth_min_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Depth Maximum must be greater than Depth Minimum.",
            )
            return
        self.action = action
        self.accept()

    @staticmethod
    def _yaml_path(path: Path) -> Path:
        if path.suffix.lower() in (".yaml", ".yml"):
            return path
        return path.with_suffix(".yaml")


class YamlConfigEditorDialog(QDialog):
    """Edit YAML source text while preserving comments and formatting."""

    def __init__(self, title: str, config_path: Path, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.saved_path: Optional[Path] = None
        self.setWindowTitle(title)
        self.resize(780, 600)
        self._setup_ui()
        self._load_path(config_path)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(str(self.config_path))
        self.path_edit.setReadOnly(True)
        self.path_edit.setCursorPosition(0)
        browse_button = QPushButton("Browse...")
        reload_button = QPushButton("Reload")
        browse_button.clicked.connect(self._browse)
        reload_button.clicked.connect(lambda: self._load_path(self.config_path))
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_button)
        path_layout.addWidget(reload_button)
        layout.addLayout(path_layout)

        self.editor = QPlainTextEdit()
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.editor, 1)

        buttons = QDialogButtonBox()
        save_button = buttons.addButton("Save", QDialogButtonBox.AcceptRole)
        save_as_button = buttons.addButton("Save As...", QDialogButtonBox.ActionRole)
        close_button = buttons.addButton(QDialogButtonBox.Close)
        save_button.clicked.connect(self._save)
        save_as_button.clicked.connect(self._save_as)
        close_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YAML Config",
            str(self.config_path.parent),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            self._validate_yaml(text)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        self.config_path = path
        self.path_edit.setText(str(path))
        self.path_edit.setCursorPosition(0)
        self.editor.setPlainText(text)

    def _save(self) -> None:
        self._write_path(self.config_path)

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save YAML Config As",
            str(self.config_path),
            "YAML Files (*.yaml *.yml)",
        )
        if path:
            self._write_path(self._yaml_path(Path(path)))

    def _write_path(self, path: Path) -> None:
        text = self.editor.toPlainText()
        try:
            self._validate_yaml(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        self.config_path = path
        self.saved_path = path
        self.accept()

    def _validate_yaml(self, text: str) -> None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("The YAML root must be a mapping.")

    @staticmethod
    def _yaml_path(path: Path) -> Path:
        if path.suffix.lower() in (".yaml", ".yml"):
            return path
        return path.with_suffix(".yaml")
