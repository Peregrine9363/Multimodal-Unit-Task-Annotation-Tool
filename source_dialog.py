# source_dialog.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Unified standard file or folder selection dialog
# ==============================================================================

from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from media_sources import IMAGE_EXTENSIONS


class DataSourceDialog(QFileDialog):
    """Select a supported file or folder with the standard Qt file dialog."""

    def __init__(
        self,
        initial_directory: Path,
        supported_extensions: Sequence[str],
        parent=None,
    ):
        super().__init__(parent, "Select Dataset File or Folder")
        self.supported_extensions = tuple(
            extension.lower() for extension in supported_extensions
        )
        self.selected_path: Optional[Path] = None

        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setFileMode(QFileDialog.AnyFile)
        self.setDirectory(str(self._initial_directory(initial_directory)))
        self.setNameFilters(self._name_filters())
        self.selectNameFilter(self._name_filters()[0])

    def accept(self) -> None:
        """Accept either the highlighted folder or a supported file."""
        selected_files = self.selectedFiles()
        if not selected_files:
            return
        selected_path = Path(selected_files[0]).expanduser()
        if selected_path.is_dir() or self._is_supported_file(selected_path):
            self.selected_path = selected_path.resolve()
            QDialog.accept(self)
            return
        QMessageBox.warning(
            self,
            "Import Data",
            "Select a supported file or folder.",
        )

    def _initial_directory(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            return resolved.parent
        if resolved.is_dir():
            return resolved
        return Path.home()

    def _name_filters(self) -> list:
        patterns = " ".join(
            f"*{extension}" for extension in self.supported_extensions
        )
        image_patterns = " ".join(
            f"*{extension}" for extension in IMAGE_EXTENSIONS
        )
        return [
            f"Supported Files ({patterns})",
            "Video Files (*.mp4)",
            f"Image Files ({image_patterns})",
            "HDF5 Files (*.h5 *.hdf5)",
            "MCAP Files (*.mcap)",
        ]

    def _is_supported_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.supported_extensions
