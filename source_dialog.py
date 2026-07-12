# source_dialog.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Unified data file or image-folder selection dialog
# ==============================================================================

from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtCore import QDir, QModelIndex, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileSystemModel,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QMessageBox,
    QPushButton,
    QStyle,
    QVBoxLayout,
)


class DataSourceDialog(QDialog):
    """Select either one supported file or one image-sequence folder."""

    def __init__(
        self,
        initial_directory: Path,
        supported_extensions: Sequence[str],
        parent=None,
    ):
        super().__init__(parent)
        self.supported_extensions = tuple(
            extension.lower() for extension in supported_extensions
        )
        self.selected_path: Optional[Path] = None
        self.current_directory = self._initial_directory(initial_directory)
        self.setWindowTitle("Import Data")
        self.resize(820, 560)
        self._setup_ui()
        self._set_directory(self.current_directory)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        path_layout = QHBoxLayout()
        self.up_button = QPushButton()
        self.up_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.up_button.setToolTip("Parent folder")
        self.path_edit = QLineEdit()
        self.up_button.clicked.connect(self._go_up)
        self.path_edit.returnPressed.connect(self._open_typed_path)
        path_layout.addWidget(self.up_button)
        path_layout.addWidget(self.path_edit, 1)
        layout.addLayout(path_layout)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        filters = [f"*{extension}" for extension in self.supported_extensions]
        self.model.setNameFilters(filters)
        self.model.setNameFilterDisables(False)
        self.model.setRootPath(str(self.current_directory))

        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.list_view.doubleClicked.connect(self._open_index)
        self.list_view.selectionModel().selectionChanged.connect(
            self._update_selected_path
        )
        layout.addWidget(self.list_view, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Open | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _initial_directory(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            return resolved.parent
        if resolved.is_dir():
            return resolved
        return Path.home()

    def _set_directory(self, directory: Path) -> None:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            return
        self.current_directory = directory
        self.path_edit.setText(str(directory))
        self.path_edit.setCursorPosition(0)
        self.list_view.setRootIndex(self.model.index(str(directory)))
        self.list_view.clearSelection()

    def _go_up(self) -> None:
        self._set_directory(self.current_directory.parent)

    def _open_typed_path(self) -> None:
        path = Path(self.path_edit.text()).expanduser()
        if path.is_dir():
            self._set_directory(path)
            return
        if self._is_supported_file(path):
            self.selected_path = path.resolve()
            self.accept()
            return
        QMessageBox.warning(self, "Import Data", "Select a supported file or folder.")

    def _open_index(self, index: QModelIndex) -> None:
        path = Path(self.model.filePath(index))
        if path.is_dir():
            self._set_directory(path)
            return
        if self._is_supported_file(path):
            self.selected_path = path.resolve()
            self.accept()

    def _update_selected_path(self) -> None:
        indexes = self.list_view.selectedIndexes()
        if not indexes:
            self.path_edit.setText(str(self.current_directory))
            return
        path = Path(self.model.filePath(indexes[0]))
        self.path_edit.setText(str(path))
        self.path_edit.setCursorPosition(0)

    def _accept_selection(self) -> None:
        indexes = self.list_view.selectedIndexes()
        path = (
            Path(self.model.filePath(indexes[0]))
            if indexes
            else self.current_directory
        )
        if path.is_dir() or self._is_supported_file(path):
            self.selected_path = path.resolve()
            self.accept()
            return
        QMessageBox.warning(self, "Import Data", "Select a supported file or folder.")

    def _is_supported_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.supported_extensions
