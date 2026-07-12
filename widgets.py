# widgets.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Reusable PyQt widgets for the dataset processor
# ==============================================================================

import os
from html import escape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/il_dataset_processor_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from data_loader import decode_depth_image_bytes, decode_image_bytes
from data_models import DataStream


class TimelineSlider(QSlider):
    """Horizontal timeline slider with click-to-seek behavior."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.setSliderDown(True)
            self._set_value_from_x(event.pos().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.isSliderDown():
            self._set_value_from_x(event.pos().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isSliderDown():
            self._set_value_from_x(event.pos().x())
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _set_value_from_x(self, x_pos: int) -> None:
        ratio = max(0.0, min(1.0, x_pos / max(self.width() - 1, 1)))
        value = self.minimum() + round((self.maximum() - self.minimum()) * ratio)
        self.setValue(max(self.minimum(), min(self.maximum(), value)))


class DataViewWidget(QWidget):
    """Display one data stream as image, graph, or text."""

    # ======================================================================
    # Depth preview visualization
    # ======================================================================
    DEPTH_COLORMAPS = {
        "grayscale": None,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
    }
    DEFAULT_DEPTH_VISUALIZATION = {
        "enabled": True,
        "colormap": "jet",
        "range_mode": "auto",
        "min_value": 0.0,
        "max_value": 2000.0,
        "contrast_power": 1.0,
        "invalid_value": 0.0,
        "invalid_color": [0, 0, 0],
        "stream_name_tokens": ["depth", "disparity"],
        "source_type_tokens": ["16uc1", "32fc1", "mono16"],
    }
    BACKGROUND_COLORS = {
        "Dark": {
            "background": "#1A1A1A",
            "text": "#EFE6D8",
            "muted": "#C9B99A",
            "border": "#8F806A",
            "grid": "#C9B99A",
            "legend": "#252525",
            "cursor": "#FFE66D",
        },
        "Light": {
            "background": "#FFF6E8",
            "text": "#3A352F",
            "muted": "#6E5F4B",
            "border": "#8F806A",
            "grid": "#C9C0B2",
            "legend": "#F4E9D8",
            "cursor": "#E05A1A",
        },
    }
    PLOT_LINE_WIDTH = 1.0
    CURSOR_LINE_WIDTH = 1.2

    def __init__(
        self,
        title: str,
        time_seek_callback: Optional[Callable[[float], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.title = title
        self.time_seek_callback = time_seek_callback
        self.stream: Optional[DataStream] = None
        self.origin_sec = 0.0
        self.default_range_sec = 10.0
        self.current_timestamp_sec = 0.0
        self.image_scale_mode = Qt.KeepAspectRatio
        self.image_zoom_factor = 1.0
        self.mouse_zoom_enabled = True
        self.mouse_tooltip_enabled = True
        self.mouse_zoom_step = 1.15
        self.tooltip_precision = 6
        self.tooltip_config = self._default_tooltip_config()
        self.depth_visualization_config = dict(self.DEFAULT_DEPTH_VISUALIZATION)
        self.plot_navigation_mode = "global_cursor"
        self._is_dragging_timestamp = False
        self._drag_start_canvas_x: Optional[float] = None
        self._drag_start_timestamp_sec: Optional[float] = None
        self._drag_seconds_per_pixel = 0.0
        self.text_zoom_point_size = 10
        self._current_image_rgb: Optional[np.ndarray] = None
        self._current_depth_image: Optional[np.ndarray] = None
        self._current_depth_range: Optional[tuple[float, float]] = None
        self._current_image_index = 0
        self.background_mode = "Dark"
        self.colors = self.BACKGROUND_COLORS[self.background_mode]
        self._cursor_line = None
        self.legend_collapsed = False
        self._collapsed_legend_patch = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("DataViewWidget")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.stack = QStackedWidget()
        self.stack.setObjectName("DataViewStack")
        self.placeholder = QLabel(self.title)
        self.placeholder.setObjectName("DataPlaceholder")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setObjectName("DataImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.text_view = QTextEdit()
        self.text_view.setObjectName("DataTextView")
        self.text_view.setReadOnly(True)
        self.text_view.setMouseTracking(True)
        self.text_view.installEventFilter(self)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.set_facecolor(self.colors["background"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("DataPlotCanvas")
        self.canvas.setMouseTracking(True)
        self.axis = self.figure.add_subplot(111)
        self._style_axis()
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_button_release)
        self.canvas.mpl_connect("figure_leave_event", lambda event: QToolTip.hideText())
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.image_label)
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.text_view)
        layout.addWidget(self.stack)
        self.set_data_view_background(self.background_mode)

    def set_data_view_background(self, mode_name: str) -> None:
        """Set the data view surface to a light or dark background."""
        self.background_mode = "Light" if mode_name == "Light" else "Dark"
        self.colors = self.BACKGROUND_COLORS[self.background_mode]
        self.setStyleSheet(self._background_stylesheet())
        self.figure.set_facecolor(self.colors["background"])
        if self.axis is not None:
            self._style_axis()
            self._style_legend()
            self._style_plot_lines()
            self._style_cursor_line()
            self._apply_legend_state()
        self.canvas.draw_idle()

    def _background_stylesheet(self) -> str:
        return f"""
        QWidget#DataViewWidget,
        QStackedWidget#DataViewStack,
        QLabel#DataPlaceholder,
        QLabel#DataImageLabel,
        QWidget#DataPlotCanvas,
        QTextEdit#DataTextView {{
            background-color: {self.colors["background"]};
            color: {self.colors["text"]};
            border: 1px solid {self.colors["border"]};
            border-radius: 2px;
        }}
        QLabel#DataPlaceholder {{
            color: {self.colors["muted"]};
            font-size: 15px;
            font-weight: bold;
        }}
        QTextEdit#DataTextView {{
            selection-background-color: {self.colors["border"]};
            selection-color: #FFFFFF;
        }}
        """

    def set_image_scale_mode(self, mode_name: str) -> None:
        """Set image scaling mode for the current view."""
        if mode_name == "Stretch":
            self.image_scale_mode = Qt.IgnoreAspectRatio
        else:
            self.image_scale_mode = Qt.KeepAspectRatio
        if self.stream is not None and self.stream.stream_type == "image":
            self._update_image(self.current_timestamp_sec)

    def set_interaction_options(
        self,
        zoom_enabled: bool,
        tooltip_enabled: bool,
        zoom_step: float,
        precision: int,
        tooltip_config: Optional[dict] = None,
    ) -> None:
        """Set mouse interaction behavior for this data view."""
        self.mouse_zoom_enabled = zoom_enabled
        self.mouse_tooltip_enabled = tooltip_enabled
        self.mouse_zoom_step = max(1.01, float(zoom_step))
        self.tooltip_precision = max(1, min(int(precision), 12))
        self.tooltip_config = tooltip_config or self._default_tooltip_config()
        self.tooltip_precision = int(
            self.tooltip_config.get("precision", self.tooltip_precision)
        )

    def set_depth_visualization(self, config: Optional[dict]) -> None:
        """Set preview-only rendering options for depth image streams."""
        merged = dict(self.DEFAULT_DEPTH_VISUALIZATION)
        if isinstance(config, dict):
            merged.update(config)
        merged["colormap"] = str(merged.get("colormap", "jet")).strip().lower()
        if merged["colormap"] not in self.DEPTH_COLORMAPS:
            merged["colormap"] = "jet"
        merged["range_mode"] = str(merged.get("range_mode", "auto")).strip().lower()
        if merged["range_mode"] not in ("auto", "manual"):
            merged["range_mode"] = "auto"
        self.depth_visualization_config = merged
        if self.stream is not None and self.stream.stream_type == "image":
            self._update_image(self.current_timestamp_sec)

    def set_plot_navigation_mode(self, mode_name: str) -> None:
        """Set how timeseries plots track the current timestamp."""
        self.plot_navigation_mode = (
            "follow_window"
            if mode_name == "follow_window"
            else "global_cursor"
        )
        self.update_timestamp(self.current_timestamp_sec)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.image_label and self.stream is not None:
            if self.stream.stream_type == "image" and event.type() == QEvent.Wheel:
                self._on_image_wheel(event)
                return True
            if self.stream.stream_type == "image" and event.type() == QEvent.MouseMove:
                self._show_image_tooltip(event.pos())
                return False
            if event.type() == QEvent.Leave:
                QToolTip.hideText()
        if watched is self.text_view and self.stream is not None:
            if self.stream.stream_type == "text" and event.type() == QEvent.Wheel:
                self._on_text_wheel(event)
                return True
            if self.stream.stream_type == "text" and event.type() == QEvent.MouseMove:
                self._show_text_tooltip(event.pos())
                return False
            if event.type() == QEvent.Leave:
                QToolTip.hideText()
        return super().eventFilter(watched, event)

    def set_stream(self, stream: Optional[DataStream], origin_sec: float) -> None:
        self.stream = stream
        self.origin_sec = origin_sec
        if stream is None or stream.is_empty():
            self.show_placeholder(self.title)
            return
        if stream.stream_type == "image":
            self.stack.setCurrentWidget(self.image_label)
            self.update_timestamp(stream.timestamps[0])
        elif stream.stream_type == "timeseries":
            self._plot_timeseries(stream)
            self.stack.setCurrentWidget(self.canvas)
            self.update_timestamp(stream.timestamps[0])
        elif stream.stream_type == "text":
            text = stream.labels[0] if stream.labels else ""
            self.text_view.setPlainText(text)
            self.stack.setCurrentWidget(self.text_view)
        else:
            self.show_placeholder(stream.name)

    def show_placeholder(self, text: str) -> None:
        self.placeholder.setText(text)
        self.stack.setCurrentWidget(self.placeholder)

    def update_timestamp(self, timestamp_sec: float) -> None:
        self.current_timestamp_sec = timestamp_sec
        if self.stream is None or self.stream.is_empty():
            return
        if self.stream.stream_type == "image":
            self._update_image(timestamp_sec)
        elif self.stream.stream_type == "timeseries" and self._cursor_line:
            self._cursor_line.set_xdata([timestamp_sec - self.origin_sec])
            if self.plot_navigation_mode == "follow_window":
                half_range = max(self.default_range_sec, 0.001) / 2.0
                center = timestamp_sec - self.origin_sec
                self.axis.set_xlim(center - half_range, center + half_range)
            self.canvas.draw_idle()

    def set_full_range(self) -> None:
        if self.stream is None or self.stream.stream_type != "timeseries":
            return
        x_values = self.stream.timestamps - self.origin_sec
        if len(x_values):
            self.axis.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
            self.canvas.draw_idle()

    def set_zoom_range(self, start_sec: float, end_sec: float) -> None:
        if self.stream is None or self.stream.stream_type != "timeseries":
            return
        self.axis.set_xlim(start_sec - self.origin_sec, end_sec - self.origin_sec)
        self.canvas.draw_idle()

    def _plot_timeseries(self, stream: DataStream) -> None:
        self.figure.clear()
        self.figure.set_facecolor(self.colors["background"])
        self.axis = self.figure.add_subplot(111)
        self._style_axis()
        x_values = stream.timestamps - self.origin_sec
        values = np.asarray(stream.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        for col_idx in range(min(values.shape[1], 8)):
            label = (
                stream.labels[col_idx]
                if col_idx < len(stream.labels)
                else f"value_{col_idx}"
            )
            self.axis.plot(
                x_values,
                values[:, col_idx],
                linewidth=self.PLOT_LINE_WIDTH,
                label=label,
            )
        self._cursor_line = self.axis.axvline(
            x_values[0],
            color=self.colors["cursor"],
            linewidth=self.CURSOR_LINE_WIDTH,
        )
        self.axis.set_title(stream.name, fontsize=9)
        self.axis.set_xlabel("Time [s]")
        self._style_axis()
        self.axis.grid(True, alpha=0.22, color=self.colors["grid"])
        legend = self.axis.legend(loc="upper right", fontsize=7)
        self._style_legend(legend)
        self.figure.tight_layout()
        self._apply_legend_state()
        self.canvas.draw_idle()

    def _style_axis(self) -> None:
        self.axis.set_facecolor(self.colors["background"])
        self.axis.tick_params(colors=self.colors["text"], labelsize=8)
        self.axis.xaxis.label.set_color(self.colors["text"])
        self.axis.yaxis.label.set_color(self.colors["text"])
        self.axis.title.set_color(self.colors["text"])
        for spine in self.axis.spines.values():
            spine.set_color(self.colors["border"])

    def _style_legend(self, legend=None) -> None:
        legend = legend or self.axis.get_legend()
        if legend is None:
            return
        legend.get_frame().set_facecolor(self.colors["legend"])
        legend.get_frame().set_edgecolor(self.colors["border"])
        for text in legend.get_texts():
            text.set_color(self.colors["text"])

    def _style_plot_lines(self) -> None:
        for line in self.axis.lines:
            if line is self._cursor_line:
                continue
            line.set_linewidth(self.PLOT_LINE_WIDTH)

    def _style_cursor_line(self) -> None:
        if self._cursor_line is not None:
            self._cursor_line.set_color(self.colors["cursor"])
            self._cursor_line.set_linewidth(self.CURSOR_LINE_WIDTH)

    def _apply_legend_state(self) -> None:
        legend = self.axis.get_legend()
        self._clear_collapsed_legend_patch()
        if legend is None:
            return
        legend.set_visible(not self.legend_collapsed)
        if self.legend_collapsed:
            self._collapsed_legend_patch = Rectangle(
                (0.962, 0.925),
                0.032,
                0.052,
                transform=self.axis.transAxes,
                facecolor=self.colors["legend"],
                edgecolor=self.colors["border"],
                linewidth=1.0,
                zorder=20,
                clip_on=False,
            )
            self.axis.add_patch(self._collapsed_legend_patch)

    def _clear_collapsed_legend_patch(self) -> None:
        if self._collapsed_legend_patch is None:
            return
        try:
            self._collapsed_legend_patch.remove()
        except (NotImplementedError, ValueError):
            pass
        self._collapsed_legend_patch = None

    def _update_image(self, timestamp_sec: float) -> None:
        if self.stream is None:
            return
        index = self.stream.nearest_index(timestamp_sec)
        if index >= len(self.stream.image_bytes):
            return
        try:
            payload = self.stream.image_bytes[index]
            image_rgb = self._render_image_payload(payload)
        except (IndexError, OSError, ValueError) as exc:
            self.image_label.setText(f"Image decode failed.\n{exc}")
            return
        if image_rgb is None:
            self.image_label.setText("Image decode failed.")
            return
        self._current_image_rgb = image_rgb
        self._current_image_index = index
        pixmap = self._numpy_to_pixmap(image_rgb)
        label_size = self.image_label.size()
        scaled_width = max(1, int(label_size.width() * self.image_zoom_factor))
        scaled_height = max(1, int(label_size.height() * self.image_zoom_factor))
        scaled = pixmap.scaled(
            scaled_width,
            scaled_height,
            self.image_scale_mode,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _render_image_payload(self, payload: bytes) -> Optional[np.ndarray]:
        """Decode an image payload without modifying the stored source bytes."""
        self._current_depth_image = None
        self._current_depth_range = None
        if self._should_render_as_depth():
            depth_image = decode_depth_image_bytes(payload)
            if depth_image is not None and depth_image.ndim == 2:
                self._current_depth_image = depth_image
                return self._depth_to_rgb(depth_image)
        return decode_image_bytes(payload)

    def _should_render_as_depth(self) -> bool:
        """Identify depth streams from configured name and source-type tokens."""
        if self.stream is None:
            return False
        if not bool(self.depth_visualization_config.get("enabled", True)):
            return False
        name = self.stream.name.lower()
        source_type = self.stream.source_type.lower()
        name_tokens = self._depth_detection_tokens("stream_name_tokens")
        source_tokens = self._depth_detection_tokens("source_type_tokens")
        return (
            any(token in name for token in name_tokens)
            or any(token in source_type for token in source_tokens)
        )

    def _depth_detection_tokens(self, config_key: str) -> List[str]:
        """Return non-empty, normalized depth detection tokens."""
        tokens = self.depth_visualization_config.get(config_key, [])
        if not isinstance(tokens, (list, tuple)):
            return []
        return [str(token).strip().lower() for token in tokens if str(token).strip()]

    def _depth_to_rgb(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert native depth values to an RGB preview image only."""
        values = np.asarray(depth_image, dtype=np.float32)
        invalid_value = float(self.depth_visualization_config.get("invalid_value", 0.0))
        valid = np.isfinite(values) & (values != invalid_value)
        min_value, max_value = self._depth_display_range(values, valid)
        self._current_depth_range = (min_value, max_value)

        normalized = np.zeros(values.shape, dtype=np.float32)
        normalized[valid] = np.clip(
            (values[valid] - min_value) / (max_value - min_value),
            0.0,
            1.0,
        )
        contrast_power = max(
            0.05,
            float(self.depth_visualization_config.get("contrast_power", 1.0)),
        )
        image_u8 = np.asarray(
            np.power(normalized, contrast_power) * 255.0,
            dtype=np.uint8,
        )
        rgb = self._apply_depth_colormap(image_u8)
        rgb[~valid] = self._invalid_depth_color()
        return rgb

    def _depth_display_range(
        self,
        values: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[float, float]:
        """Resolve the manual or per-frame automatic display range."""
        range_mode = str(self.depth_visualization_config.get("range_mode", "auto"))
        if range_mode == "manual":
            min_value = float(self.depth_visualization_config.get("min_value", 0.0))
            max_value = float(self.depth_visualization_config.get("max_value", 1.0))
        elif np.any(valid):
            min_value = float(np.min(values[valid]))
            max_value = float(np.max(values[valid]))
        else:
            min_value, max_value = 0.0, 1.0
        if max_value <= min_value:
            max_value = min_value + 1.0
        return min_value, max_value

    def _apply_depth_colormap(self, image_u8: np.ndarray) -> np.ndarray:
        """Apply the configured OpenCV colormap and return RGB pixels."""
        colormap_name = str(
            self.depth_visualization_config.get("colormap", "jet")
        ).lower()
        colormap = self.DEPTH_COLORMAPS.get(colormap_name)
        if colormap is None:
            return np.repeat(image_u8[:, :, None], 3, axis=2)
        image_bgr = cv2.applyColorMap(image_u8, colormap)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _invalid_depth_color(self) -> np.ndarray:
        """Return the configured RGB color for invalid depth pixels."""
        color = self.depth_visualization_config.get("invalid_color", [0, 0, 0])
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            color = [0, 0, 0]
        return np.asarray(
            [int(np.clip(value, 0, 255)) for value in color[:3]],
            dtype=np.uint8,
        )

    def _on_image_wheel(self, event) -> None:
        if not self.mouse_zoom_enabled:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.mouse_zoom_step if delta > 0 else 1.0 / self.mouse_zoom_step
        self.image_zoom_factor = float(np.clip(self.image_zoom_factor * factor, 0.25, 8.0))
        self._update_image(self.current_timestamp_sec)
        event.accept()

    def _show_image_tooltip(self, pos: QPoint) -> None:
        if not self.mouse_tooltip_enabled or self.stream is None or self._current_image_rgb is None:
            return
        image_pos = self._label_pos_to_image_pos(pos)
        index = self._current_image_index
        timestamp = float(self.stream.timestamps[index])
        rel_time = timestamp - self.origin_sec
        lines = []
        if self._tooltip_enabled("data_type", "image"):
            lines.append("Data type: image")
        if self._tooltip_enabled("topic", "image"):
            lines.append(f"Topic: {self.stream.name}")
        if self._tooltip_enabled("time", "image") or self._tooltip_enabled("cursor", "image"):
            lines.append(f"Time: {self._format_float(rel_time)} s")
        if self._tooltip_enabled("frame", "image") or self._tooltip_enabled("cursor", "image"):
            lines.append(f"Frame: {index + 1}/{len(self.stream.timestamps)}")
        if self._tooltip_enabled("zoom", "image") or self._tooltip_enabled("cursor", "image"):
            lines.append(f"Zoom: {self.image_zoom_factor:.2f}x")
        if self._tooltip_enabled("timestamp", "image"):
            lines.append(f"Timestamp: {timestamp:.9f}")
        if (
            self._tooltip_enabled("pixel", "image")
            or self._tooltip_enabled("value", "image")
        ) and image_pos is not None:
            x_pos, y_pos = image_pos
            rgb = self._current_image_rgb[y_pos, x_pos]
            if self._current_depth_image is not None:
                depth_value = self._current_depth_image[y_pos, x_pos]
                range_text = ""
                if self._current_depth_range is not None:
                    min_value, max_value = self._current_depth_range
                    range_text = f", range={min_value:.6g}..{max_value:.6g}"
                lines.append(
                    "Pixel: "
                    f"x={x_pos}, y={y_pos}, depth={float(depth_value):.6g}, "
                    f"RGB={tuple(int(v) for v in rgb[:3])}{range_text}"
                )
            else:
                lines.append(
                    f"Pixel: x={x_pos}, y={y_pos}, "
                    f"RGB={tuple(int(v) for v in rgb[:3])}"
                )
        self._show_rich_tooltip(self.image_label, pos, lines)

    def _label_pos_to_image_pos(self, pos: QPoint) -> Optional[tuple[int, int]]:
        pixmap = self.image_label.pixmap()
        image = self._current_image_rgb
        if pixmap is None or image is None:
            return None
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        left = (self.image_label.width() - pixmap_width) / 2.0
        top = (self.image_label.height() - pixmap_height) / 2.0
        x_scaled = pos.x() - left
        y_scaled = pos.y() - top
        if x_scaled < 0 or y_scaled < 0 or x_scaled >= pixmap_width or y_scaled >= pixmap_height:
            return None
        height, width = image.shape[:2]
        x_pos = int(np.clip(x_scaled * width / max(pixmap_width, 1), 0, width - 1))
        y_pos = int(np.clip(y_scaled * height / max(pixmap_height, 1), 0, height - 1))
        return x_pos, y_pos

    def _on_plot_scroll(self, event) -> None:
        if not self.mouse_zoom_enabled:
            return
        if self.stream is None or self.stream.stream_type != "timeseries" or event.inaxes != self.axis:
            return
        if event.xdata is None:
            return
        left, right = self.axis.get_xlim()
        current_width = max(right - left, 1e-9)
        factor = 1.0 / self.mouse_zoom_step if event.button == "up" else self.mouse_zoom_step
        new_width = max(current_width * factor, 1e-6)
        center = float(event.xdata)
        ratio = (center - left) / current_width
        new_left = center - new_width * ratio
        new_right = new_left + new_width
        self.axis.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def _on_plot_motion(self, event) -> None:
        if self.stream is None or self.stream.stream_type != "timeseries" or event.inaxes != self.axis:
            QToolTip.hideText()
            return
        if event.xdata is None:
            return
        if self._is_dragging_timestamp:
            self._seek_from_plot_drag(event)
            return
        if not self.mouse_tooltip_enabled:
            return
        timestamp = self.origin_sec + float(event.xdata)
        index = self.stream.nearest_index(timestamp)
        tooltip = self._timeseries_tooltip(index, timestamp, event.ydata)
        qt_y = self.canvas.height() - int(event.y)
        QToolTip.showText(
            self.canvas.mapToGlobal(QPoint(int(event.x) + 14, qt_y + 14)),
            tooltip,
        )

    def _on_plot_button_press(self, event) -> None:
        if not self._is_left_plot_button(event.button):
            return
        if self._handle_legend_toggle_click(event):
            return
        if event.inaxes != self.axis or event.xdata is None:
            return
        if self.stream is None or self.stream.stream_type != "timeseries":
            return
        self._is_dragging_timestamp = True
        self._start_plot_drag(event)
        if self.plot_navigation_mode != "follow_window":
            self._seek_to_plot_timestamp(event.xdata)

    def _on_plot_button_release(self, event) -> None:
        self._is_dragging_timestamp = False
        self._drag_start_canvas_x = None
        self._drag_start_timestamp_sec = None
        self._drag_seconds_per_pixel = 0.0

    def _start_plot_drag(self, event) -> None:
        self._drag_start_canvas_x = float(event.x)
        self._drag_start_timestamp_sec = self.current_timestamp_sec
        left, right = self.axis.get_xlim()
        width_pixels = max(float(self.canvas.width()), 1.0)
        self._drag_seconds_per_pixel = float(right - left) / width_pixels

    def _seek_from_plot_drag(self, event) -> None:
        if self.plot_navigation_mode != "follow_window":
            self._seek_to_plot_timestamp(event.xdata)
            return
        if self._drag_start_canvas_x is None or self._drag_start_timestamp_sec is None:
            return
        delta_pixels = float(event.x) - self._drag_start_canvas_x
        target_sec = self._drag_start_timestamp_sec - delta_pixels * self._drag_seconds_per_pixel
        self._seek_to_absolute_timestamp(target_sec)

    def _seek_to_plot_timestamp(self, relative_timestamp: float) -> None:
        self._seek_to_absolute_timestamp(self.origin_sec + float(relative_timestamp))

    def _seek_to_absolute_timestamp(self, timestamp_sec: float) -> None:
        if self.time_seek_callback is None:
            return
        self.time_seek_callback(float(timestamp_sec))

    def _handle_legend_toggle_click(self, event) -> bool:
        if self.stream is None or self.stream.stream_type != "timeseries":
            return False
        if self.legend_collapsed:
            if not self._is_collapsed_legend_hit(event):
                return False
            self._set_legend_collapsed(False)
            return True
        if not self._is_expanded_legend_hit(event):
            return False
        self._set_legend_collapsed(True)
        return True

    def _set_legend_collapsed(self, collapsed: bool) -> None:
        self.legend_collapsed = collapsed
        self._apply_legend_state()
        QToolTip.hideText()
        self.canvas.draw_idle()

    def _is_expanded_legend_hit(self, event) -> bool:
        legend = self.axis.get_legend()
        if legend is None or not legend.get_visible():
            return False
        renderer = self.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
        return bool(bbox.contains(float(event.x), float(event.y)))

    def _is_collapsed_legend_hit(self, event) -> bool:
        if self._collapsed_legend_patch is None:
            return False
        x_min, y_min = self.axis.transAxes.transform((0.962, 0.925))
        x_max, y_max = self.axis.transAxes.transform((0.994, 0.977))
        left, right = sorted((x_min, x_max))
        bottom, top = sorted((y_min, y_max))
        return left <= float(event.x) <= right and bottom <= float(event.y) <= top

    def _is_left_plot_button(self, button) -> bool:
        if button in (1, "button1"):
            return True
        return getattr(button, "name", "") == "LEFT"

    def _timeseries_tooltip(
        self,
        index: int,
        cursor_timestamp: float,
        cursor_value: Optional[float] = None,
    ) -> str:
        if self.stream is None or self.stream.values is None:
            return ""
        sample_timestamp = float(self.stream.timestamps[index])
        values = np.asarray(self.stream.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        lines = []
        if self._tooltip_enabled("data_type", "timeseries"):
            lines.append("Data type: timeseries")
        if self._tooltip_enabled("topic", "timeseries"):
            lines.append(f"Topic: {self.stream.name}")
        if self._tooltip_enabled("cursor", "timeseries"):
            lines.append(f"Cursor time: {self._format_float(cursor_timestamp - self.origin_sec)} s")
            if cursor_value is not None:
                lines.append(("Cursor value", self._format_float(cursor_value), cursor_value))
        if self._tooltip_enabled("sample_time", "timeseries"):
            lines.append(f"Sample time: {self._format_float(sample_timestamp - self.origin_sec)} s")
        if self._tooltip_enabled("sample", "timeseries"):
            lines.append(f"Sample: {index + 1}/{len(self.stream.timestamps)}")
        if self._tooltip_enabled("timestamp", "timeseries"):
            lines.append(f"Timestamp: {sample_timestamp:.9f}")
        if self._tooltip_enabled("values", "timeseries") or self._tooltip_enabled("value", "timeseries"):
            for col_idx in range(min(values.shape[1], 8)):
                label = (
                    self.stream.labels[col_idx]
                    if col_idx < len(self.stream.labels)
                    else f"value_{col_idx}"
                )
                value = float(values[index, col_idx])
                lines.append((label, self._format_float(value), value))
        return self._tooltip_html(lines)

    def _on_text_wheel(self, event) -> None:
        if not self.mouse_zoom_enabled:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.text_zoom_point_size += 1 if delta > 0 else -1
        self.text_zoom_point_size = int(np.clip(self.text_zoom_point_size, 6, 32))
        font = self.text_view.font()
        font.setPointSize(self.text_zoom_point_size)
        self.text_view.setFont(font)
        event.accept()

    def _show_text_tooltip(self, pos: QPoint) -> None:
        if not self.mouse_tooltip_enabled or self.stream is None:
            return
        cursor = self.text_view.cursorForPosition(pos)
        text = self.text_view.toPlainText()
        char_index = cursor.position()
        lines = []
        if self._tooltip_enabled("data_type", "text"):
            lines.append("Data type: text")
        if self._tooltip_enabled("topic", "text"):
            lines.append(f"Topic: {self.stream.name}")
        if self._tooltip_enabled("timestamp", "text"):
            lines.append(f"Timestamp: {float(self.stream.timestamps[0]):.9f}")
        if self._tooltip_enabled("cursor", "text"):
            lines.append(f"Line: {cursor.blockNumber() + 1}")
            lines.append(f"Column: {cursor.positionInBlock() + 1}")
            lines.append(f"Character index: {char_index}")
            lines.append(f"Zoom: {self.text_zoom_point_size} pt")
        if self._tooltip_enabled("value", "text") and 0 <= char_index < len(text):
            lines.append(f"Character: {repr(text[char_index])}")
        self._show_rich_tooltip(self.text_view, pos, lines)

    def _tooltip_enabled(self, field_name: str, stream_type: str) -> bool:
        field_key = f"{stream_type}_fields"
        fields = self.tooltip_config.get(field_key, [])
        return field_name in set(fields)

    def _format_float(self, value: float) -> str:
        return f"{float(value):.{self.tooltip_precision}g}"

    def _default_tooltip_config(self) -> dict:
        return {
            "enabled": True,
            "precision": 6,
            "colors": {
                "positive": "#D32F2F",
                "negative": "#1565C0",
                "neutral": "#202428",
            },
            "image_fields": ["time", "frame", "zoom", "pixel"],
            "timeseries_fields": ["sample_time", "sample", "values"],
            "text_fields": ["timestamp", "cursor", "value"],
        }

    def _show_rich_tooltip(
        self,
        widget: QWidget,
        pos: QPoint,
        lines: List[Union[str, tuple]],
    ) -> None:
        QToolTip.showText(widget.mapToGlobal(pos + QPoint(14, 14)), self._tooltip_html(lines))

    def _tooltip_html(self, lines: List[Union[str, tuple]]) -> str:
        body = "<br>".join(self._tooltip_line_html(line) for line in lines)
        return f"<html><body style='white-space: nowrap;'>{body}</body></html>"

    def _tooltip_line_html(self, line: Union[str, tuple]) -> str:
        if not isinstance(line, tuple):
            return escape(str(line))
        label, value_text, numeric_value = line
        color = self._value_color(float(numeric_value))
        return (
            f"{escape(str(label))}: "
            f"<span style='color: {color}; font-weight: 600;'>"
            f"{escape(str(value_text))}</span>"
        )

    def _value_color(self, value: float) -> str:
        colors = self.tooltip_config.get("colors", {})
        if value > 0.0:
            return colors.get("positive", "#D32F2F")
        if value < 0.0:
            return colors.get("negative", "#1565C0")
        return colors.get("neutral", "#202428")

    def _numpy_to_pixmap(self, image_rgb: np.ndarray) -> QPixmap:
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.stream is not None and self.stream.stream_type == "image":
            self._update_image(self.current_timestamp_sec)


class DataDockWidget(QDockWidget):
    """Dockable view with stream selector."""

    stream_changed = None

    def __init__(
        self,
        title: str,
        change_callback: Callable[[int, str], None],
        index: int,
        time_seek_callback: Optional[Callable[[float], None]] = None,
        pop_callback: Optional[Callable[[int], None]] = None,
        parent=None,
    ):
        super().__init__(title, parent)
        self.index = index
        self.change_callback = change_callback
        self.pop_callback = pop_callback
        self.view = DataViewWidget(title, time_seek_callback)
        self.namespace_combo = QComboBox()
        self.namespace_combo.setObjectName("DataNamespaceCombo")
        self.namespace_combo.setMinimumWidth(82)
        self.namespace_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(12)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.stream_names_by_namespace: Dict[str, List[str]] = {"general": []}
        self.namespace_labels: Dict[str, str] = {"general": "General"}
        self.fit_button = QPushButton("Fit")
        self.fit_button.setObjectName("DataFitButton")
        self.fit_button.setFixedHeight(22)
        self.fit_button.setMinimumWidth(88)
        self.fit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title_button = QToolButton()
        self.title_button.setObjectName("DataTitlePopButton")
        self.title_button.setFixedSize(18, 18)
        self.title_button.setToolTip("Pop out")
        self.title_button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setTitleBarWidget(self._make_title_bar())
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(3)
        top_layout.addWidget(self.namespace_combo)
        top_layout.addWidget(self.combo, 3)
        top_layout.addWidget(self.fit_button, 1)
        layout.addLayout(top_layout)
        layout.addWidget(self.view, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWidget(container)
        self.setObjectName(f"DataDock_{self.index + 1}")
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.namespace_combo.currentIndexChanged.connect(self._on_namespace_changed)
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        self.fit_button.clicked.connect(self.view.set_full_range)
        self.title_button.clicked.connect(self._on_title_button_clicked)

    def _make_title_bar(self) -> QWidget:
        title_bar = QWidget()
        title_bar.setObjectName("DataDockTitleBar")
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(2)
        left_spacer = QWidget()
        left_spacer.setFixedSize(18, 18)
        title_label = QLabel(self.windowTitle())
        title_label.setObjectName("DataDockTitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(left_spacer)
        layout.addWidget(title_label, 1)
        layout.addWidget(self.title_button, 0, Qt.AlignRight | Qt.AlignVCenter)
        return title_bar

    def set_stream_names(self, names: List[str]) -> None:
        self.set_namespace_groups({"general": names}, {"general": "General"})

    def set_namespace_groups(
        self,
        stream_names_by_namespace: Dict[str, List[str]],
        namespace_labels: Dict[str, str],
    ) -> None:
        current_namespace = self.current_namespace()
        current_stream = self.combo.currentText()
        self.stream_names_by_namespace = {
            key: list(value)
            for key, value in stream_names_by_namespace.items()
        }
        self.namespace_labels = dict(namespace_labels)
        namespace_ids = list(self.stream_names_by_namespace.keys())
        if current_namespace not in namespace_ids:
            current_namespace = namespace_ids[0] if namespace_ids else "general"
        self.namespace_combo.blockSignals(True)
        self.combo.blockSignals(True)
        self.namespace_combo.clear()
        for namespace_id in namespace_ids:
            label = self.namespace_labels.get(namespace_id, namespace_id.title())
            self.namespace_combo.addItem(label, namespace_id)
        namespace_index = self.namespace_combo.findData(current_namespace)
        self.namespace_combo.setCurrentIndex(max(namespace_index, 0))
        self._reload_stream_combo(current_namespace, current_stream)
        self.namespace_combo.blockSignals(False)
        self.combo.blockSignals(False)

    def current_namespace(self) -> str:
        namespace_id = self.namespace_combo.currentData()
        return str(namespace_id or "general")

    def set_namespace(self, namespace_id: str) -> None:
        index = self.namespace_combo.findData(namespace_id)
        if index < 0:
            return
        self.namespace_combo.blockSignals(True)
        self.combo.blockSignals(True)
        self.namespace_combo.setCurrentIndex(index)
        self._reload_stream_combo(namespace_id, self.combo.currentText())
        self.namespace_combo.blockSignals(False)
        self.combo.blockSignals(False)

    def _reload_stream_combo(self, namespace_id: str, preferred_name: str = "") -> None:
        names = self.stream_names_by_namespace.get(namespace_id, [])
        self.combo.clear()
        self.combo.addItem("")
        self.combo.addItems(names)
        if preferred_name in names:
            self.combo.setCurrentText(preferred_name)

    def select_stream(self, name: str) -> None:
        if name:
            for namespace_id, names in self.stream_names_by_namespace.items():
                if name in names:
                    self.set_namespace(namespace_id)
                    break
        self.combo.blockSignals(True)
        self.combo.setCurrentText(name if name else "")
        self.combo.blockSignals(False)

    def _on_namespace_changed(self) -> None:
        self.combo.blockSignals(True)
        self._reload_stream_combo(self.current_namespace())
        self.combo.blockSignals(False)

    def _on_combo_changed(self, name: str) -> None:
        self.change_callback(self.index, name)

    def set_popped_out(self, is_popped_out: bool) -> None:
        """팝업/도킹 상태에 맞춰 버튼 텍스트를 갱신합니다."""
        self.title_button.setToolTip("Dock to grid" if is_popped_out else "Pop out")

    def _on_title_button_clicked(self) -> None:
        if self.pop_callback is not None:
            self.pop_callback(self.index)
