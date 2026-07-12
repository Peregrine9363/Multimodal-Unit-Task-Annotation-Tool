# progress_dialog.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Shared modal progress dialog for import and export operations
# ==============================================================================

import time
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class OperationProgressDialog(QDialog):
    """Display operation progress, elapsed time, and estimated remaining time."""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self._started_at = time.monotonic()
        self._allow_close = False
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(520)
        self._setup_ui(message)

    def _setup_ui(self, message: str) -> None:
        layout = QVBoxLayout(self)
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.timing_label = QLabel("Elapsed 00:00 | ETA estimating")
        layout.addWidget(self.message_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.timing_label)

    def update_progress(self, percent: int, message: str) -> None:
        """Update visible progress while reserving 100% for completion."""
        value = max(0, min(int(percent), 99))
        self.message_label.setText(message)
        self.progress_bar.setValue(value)
        self.timing_label.setText(self._timing_text(value))

    def finish(self, message: str = "Completed.") -> None:
        """Mark the operation complete and allow the dialog to close."""
        self._allow_close = True
        self.message_label.setText(message)
        self.progress_bar.setValue(100)
        self.timing_label.setText(self._timing_text(100))

    def allow_close(self) -> None:
        """Allow cleanup after an error without reporting false completion."""
        self._allow_close = True

    def _timing_text(self, percent: int) -> str:
        elapsed = max(0.0, time.monotonic() - self._started_at)
        eta = self._estimated_remaining(percent, elapsed)
        return (
            f"Elapsed {self._format_duration(elapsed)} | "
            f"ETA {self._format_duration(eta)}"
        )

    @staticmethod
    def _estimated_remaining(percent: int, elapsed: float) -> Optional[float]:
        if percent >= 100:
            return 0.0
        if percent <= 0 or elapsed <= 0.0:
            return None
        return elapsed * (100.0 - percent) / float(percent)

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None:
            return "estimating"
        total_seconds = int(max(0.0, seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, second = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{second:02d}"
        return f"{minutes:02d}:{second:02d}"

    def reject(self) -> None:
        if self._allow_close:
            super().reject()

    def closeEvent(self, event) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
