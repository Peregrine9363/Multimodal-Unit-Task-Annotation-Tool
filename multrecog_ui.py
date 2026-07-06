# multrecog_ui.py
import io
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from PyQt5.QtWidgets import (QSlider, QStyleOptionSlider, QStyle, QDialog, 
                             QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox,
                             QGraphicsView, QToolButton, QMenu, QAction)
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtCore import Qt, QRectF, QSizeF, QObject
import pyqtgraph as pg

class SegmentedSlider(QSlider):
    """
    설정값(config)을 받아 라벨 구간을 그리는 커스텀 슬라이더.
    """
    def __init__(self, orientation, style_config: Dict[str, Any], parent=None):
        super().__init__(orientation, parent)
        self.segments: List[Tuple[int, int, QColor]] = []
        self.preview_segment: Optional[Tuple[int, int, QColor]] = None
        self.cfg = style_config

    def set_segments(self, segments: List[Tuple[int, int, QColor]]):
        self.segments = segments
        self.update()

    def set_preview_segment(self, segment: Optional[Tuple[int, int, QColor]]):
        self.preview_segment = segment
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        groove_h = self.cfg["GROOVE_HEIGHT"]
        border_w = self.cfg["BORDER_WIDTH"]
        
        orig_groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        disp_groove = QRectF(orig_groove)
        disp_groove.setHeight(groove_h)
        disp_groove.moveCenter(orig_groove.center())

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(self.cfg["GROOVE_BORDER_COLOR"])))
        painter.drawRoundedRect(disp_groove, 2, 2)
        
        inner_groove = disp_groove.adjusted(border_w, border_w, -border_w, -border_w)
        painter.setBrush(QBrush(QColor(self.cfg["GROOVE_COLOR"])))
        painter.drawRect(inner_groove)

        for s, e, c in self.segments:
            self._draw_segment(painter, inner_groove, s, e, c)
        if self.preview_segment:
            s, e, c = self.preview_segment
            pc = QColor(c)
            pc.setAlpha(128)
            self._draw_segment(painter, inner_groove, s, e, pc)

        orig_handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        disp_handle = QRectF(orig_handle)
        disp_handle.setSize(QSizeF(self.cfg["HANDLE_WIDTH"], self.cfg["HANDLE_HEIGHT"]))
        disp_handle.moveCenter(orig_handle.center())

        painter.setBrush(QBrush(QColor(self.cfg["HANDLE_BORDER_COLOR"])))
        painter.drawRoundedRect(disp_handle, 5, 5)
        inner_handle = disp_handle.adjusted(border_w, border_w, -border_w, -border_w)
        painter.setBrush(QBrush(QColor(self.cfg["HANDLE_COLOR"])))
        painter.drawRoundedRect(inner_handle, 5, 5)

    def _draw_segment(self, painter, rect, start, end, color):
        total = self.maximum() + 1
        if total <= 1: return
        s_pos = rect.left() + (rect.width() * start / total)
        e_pos = rect.left() + (rect.width() * (end + 1) / total)
        
        # [수정] 원본과 동일하게 1픽셀을 추가하여 시각적 틈을 메움
        width = e_pos - s_pos + 1
        
        painter.setBrush(QBrush(color))
        painter.drawRect(QRectF(s_pos, rect.top(), width, rect.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            hit = self.style().hitTestComplexControl(QStyle.CC_Slider, opt, event.pos(), self)
            if hit != QStyle.SC_SliderHandle:
                groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
                pos = event.pos().x() - groove.left()
                val = self.minimum() + round((self.maximum() - self.minimum()) * (pos / groove.width()))
                self.setValue(max(self.minimum(), min(self.maximum(), val)))
        super().mousePressEvent(event)


class EditSegmentDialog(QDialog):
    def __init__(self, segment, total_frames, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Segment")
        self.total_frames = total_frames
        self.start_inp = QLineEdit(str(segment[0]))
        self.end_inp = QLineEdit(str(segment[1]))
        self.cls_inp = QLineEdit(", ".join(map(str, segment[2])))
        
        layout = QFormLayout(self)
        layout.addRow("Start:", self.start_inp)
        layout.addRow("End:", self.end_inp)
        layout.addRow("Class IDs:", self.cls_inp)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_data(self):
        try:
            s, e = int(self.start_inp.text()), int(self.end_inp.text())
            c_ids = [int(x) for x in self.cls_inp.text().replace(",", " ").split() if x]
            if not c_ids or len(c_ids) > 2: return None
            if not (0 <= s < self.total_frames and 0 <= e < self.total_frames and s <= e): return None
            return (s, e, c_ids)
        except: return None


class ViewManager:
    """
    pyqtgraph 뷰 제어 및 렌더링 관리 클래스.
    """
    def __init__(self, view_widget: QGraphicsView, tool_button: QToolButton, 
                 event_filter_target: QObject, plot_window_size: int,
                 plot_config: Dict[str, Any]):
        self.view_widget = view_widget
        self.tool_button = tool_button
        self.event_filter_target = event_filter_target
        self.plot_window_size = plot_window_size
        self.plot_cfg = plot_config
        
        self.stream_name: Optional[str] = None
        self.data: Optional[np.ndarray] = None
        self.is_compressed_image = False
        self.initial_fit_done = False

        # QSS 배경색이 반영될 수 있도록 pyqtgraph 요소들을 투명하게 설정
        layout = pg.GraphicsLayoutWidget(parent=self.view_widget)
        layout.setBackground('transparent') # 레이아웃 투명화
        layout.setGeometry(self.view_widget.rect())
        self.plot_widget = layout.addPlot()
        self.plot_widget.getViewBox().setBackgroundColor('transparent') # 뷰박스 투명화
        
        self._setup_axis_style()

    def _setup_axis_style(self):
        """축의 색상과 격자 스타일을 설정합니다."""
        for axis_name in ['left', 'bottom']:
            axis = self.plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=self.plot_cfg["AXIS_COLOR"], width=0.8))
            axis.setTextPen(pg.mkPen(color=self.plot_cfg["AXIS_COLOR"]))
        self.plot_widget.showGrid(x=True, y=True, alpha=self.plot_cfg["GRID_ALPHA"])

    def setup_menu(self, stream_names: List[str], callback):
        menu = QMenu(self.tool_button)
        menu.installEventFilter(self.event_filter_target)
        for name in stream_names:
            action = QAction(name, menu)
            action.triggered.connect(lambda c, s=name: callback(self, s))
            menu.addAction(action)
        self.tool_button.setMenu(menu)

    def assign_data(self, stream_name: str, data: np.ndarray, total_frames: int):
        self.stream_name = stream_name
        self.data = data
        self.tool_button.setText(stream_name)
        self.is_compressed_image = (data is not None and data.dtype == np.object_)
        self.initial_fit_done = False
        self._plot_data(total_frames)

    def _plot_data(self, total_frames: int):
        self.plot_widget.clear()
        self.plot_widget.setLimits(xMin=None, yMin=None, xMax=None, yMax=None)
        
        if self.data is None: return
        
        is_img = self.is_compressed_image or (self.data.ndim in [3, 4] and self.data.dtype == np.uint8)
        if is_img:
            self._setup_image_view()
        else:
            self._setup_timeseries_view(total_frames)

    def _setup_image_view(self):
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.invertY(True)
        self.plot_widget.setAspectLocked(True)
        img_item = pg.ImageItem()
        
        final_img = None
        if len(self.data) > 0:
            if self.is_compressed_image:
                final_img = self._decode_image(self.data[0])
            else:
                final_img = np.transpose(self.data[0], (1, 0, 2))
            
            if final_img is not None: img_item.setImage(final_img)

        self.plot_widget.addItem(img_item)
        if final_img is not None:
            self.fit_view_to_data()
            self.initial_fit_done = True
        else:
            self.plot_widget.autoRange()

    def _decode_image(self, raw_data):
        try:
            buf = np.frombuffer(raw_data, np.uint8)
            if buf.size == 0: return None
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None: return None
            
            try:
                pil = Image.open(io.BytesIO(raw_data))
                exif = pil._getexif()
                orient = exif.get(0x0112, 1) if exif else 1
                if orient == 2: img = cv2.flip(img, 1)
                elif orient == 3: img = cv2.rotate(img, cv2.ROTATE_180)
                elif orient == 4: img = cv2.flip(img, 0)
                elif orient == 5: img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                elif orient == 6: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif orient == 7: img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
                elif orient == 8: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            except: pass
            
            return np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1, 0, 2))
        except: return None

    def _setup_timeseries_view(self, total_frames):
        self._setup_axis_style()
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.invertY(False)
        self.plot_widget.addLegend()

        data = self.data.astype(np.float64)
        data[~np.isfinite(data)] = np.nan
        
        if data.ndim == 1:
            self.plot_widget.plot(data, pen=self.plot_cfg["LINE_COLORS"][-1], name=self.stream_name, connect='finite')
        elif data.ndim == 2:
            pens = self.plot_cfg["LINE_COLORS"]
            for i in range(data.shape[1]):
                self.plot_widget.plot(data[:, i], pen=pens[i % len(pens)], name=f"{self.stream_name}_{i}", connect='finite')

        self.plot_widget.addItem(pg.InfiniteLine(angle=90, movable=False, pen=self.plot_cfg["CURR_POS_LINE"]))
        
        valid = np.isfinite(self.data)
        if np.any(valid):
            mn, mx = np.min(self.data[valid]), np.max(self.data[valid])
            pad = (mx - mn) * 0.15 or 1.0
            self.plot_widget.setLimits(xMin=0, xMax=total_frames-1, yMin=mn-pad, yMax=mx+pad)
            self.plot_widget.setYRange(mn-pad, mx+pad)
        else:
            self.plot_widget.setLimits(xMin=0, xMax=total_frames-1, yMin=0, yMax=1)

    def update_view(self, index: int, total_frames: int):
        if self.data is None or index >= len(self.data): return
        
        is_img = self.is_compressed_image or (self.data.ndim in [3, 4] and self.data.dtype == np.uint8)
        if is_img:
            item = next((i for i in self.plot_widget.items if isinstance(i, pg.ImageItem)), None)
            if item:
                img = self._decode_image(self.data[index]) if self.is_compressed_image else np.transpose(self.data[index], (1, 0, 2))
                if img is not None:
                    item.setImage(img, autoLevels=False)
                    if not self.initial_fit_done:
                        self.fit_view_to_data()
                        self.initial_fit_done = True
        else:
            line = next((i for i in self.plot_widget.items if isinstance(i, pg.InfiniteLine)), None)
            if line:
                line.setPos(index)
                if not self.initial_fit_done:
                    self.fit_view_to_data()
                    self.initial_fit_done = True
                else:
                    sx = max(0, index - self.plot_window_size)
                    ex = min(total_frames - 1, index + self.plot_window_size)
                    self.plot_widget.setXRange(sx, ex, padding=0)

    def fit_view_to_data(self):
        item = next((i for i in self.plot_widget.items if isinstance(i, pg.ImageItem)), None)
        if item and item.image is not None:
            w, h = item.image.shape[0], item.image.shape[1]
            self.plot_widget.setRange(xRange=(0, w), yRange=(0, h), padding=0)
        else:
            self.plot_widget.autoRange()