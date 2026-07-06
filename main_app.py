# main_app.py
# ==============================================================================
# MulTRecog labeling tool main controller
# ==============================================================================

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app_config import (
    APP_TITLE,
    CLASS_COLORS,
    DEFAULT_DATA_VIEW_BACKGROUND,
    DEFAULT_HDF5_MAPPING_FILE,
    DEFAULT_LABEL_DATASET,
    DEFAULT_VIEW_CONFIG_FILE,
    DEFAULT_ZOOM_SECONDS,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    EXPORT_DIR_NAME,
    MAIN_UI_FILE,
    MAX_CLASS_IDS_PER_SEGMENT,
    PREVIEW_DATA_VIEW_COUNT,
    PREVIEW_MAX_IMAGE_FRAMES,
    PREVIEW_MAX_NUMERIC_SAMPLES,
    SLIDER_STYLE,
    STYLE_FILE,
    SUPPORTED_EXTENSIONS,
    TIMELINE_STEPS,
    TOGGLE_LABELING_KEYS,
    load_yaml,
)
from data_loader import DatasetLoader, discover_supported_files
from data_models import DatasetSession
from labeling_io import LabelImportResult, LabelStorage
from multrecog_core import LabelingLogic
from multrecog_ui import EditSegmentDialog, SegmentedSlider
from widgets import DataDockWidget


ROS2_DISTRO = "humble"
ROS2_PREFIX = Path(f"/opt/ros/{ROS2_DISTRO}")
ROS2_LOCAL_PYTHON = ROS2_PREFIX / "local/lib/python3.10/dist-packages"
ROS2_GLOBAL_PYTHON = ROS2_PREFIX / "lib/python3.10/site-packages"
ROS2_LIB_DIRS = [
    ROS2_PREFIX / "lib",
    ROS2_PREFIX / "local/lib",
]
ROS2_REEXEC_FLAG = "MULTRECOG_ROS2_ENV_BOOTSTRAPPED"
SIDE_PANEL_MIN_WIDTH = 320
SIDE_PANEL_MAX_WIDTH = 420
SIDE_PANEL_INITIAL_WIDTH = 360
TIMELINE_PANEL_MIN_HEIGHT = 70
TIMELINE_PANEL_INITIAL_HEIGHT = 90
TIMELINE_PANEL_MAX_HEIGHT = 120
TIMELINE_BUTTON_HEIGHT = 32
TIMELINE_BUTTON_WIDTHS = {
    "boundary": 40,  # <<, >> 버튼 너비.
    "frame": 60,     # <, > 버튼 너비.
    "label": 100,     # Start/Stop 버튼 너비.
}
VIEW_SETTINGS_ROW_SPACING = 6       # Data View Settings 행 사이 간격.
VIEW_SETTINGS_COLUMN_SPACING = 8    # Data View Settings 라벨/입력 간격.
VIEW_SETTINGS_BUTTON_SPACING = 6    # Data View Settings 버튼 사이 간격.
VIEW_SETTINGS_LABEL_MIN_WIDTH = 88  # Data View Settings 라벨 최소 너비.
VIEW_SETTINGS_LABEL_MAX_WIDTH = 112  # Data View Settings 라벨 최대 너비.
VIEW_SETTINGS_CONTROL_HEIGHT = 32   # Data View Settings 라벨/버튼 공통 높이.


try:
    Ui_MainWindow, _ = uic.loadUiType(str(MAIN_UI_FILE))
except FileNotFoundError:
    print(f"Error: UI file '{MAIN_UI_FILE}' not found.")
    sys.exit(1)


def load_stylesheet(path: Path) -> str:
    """QSS 파일을 문자열로 읽습니다."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Failed to load stylesheet '{path}': {exc}")
        return ""


def bootstrap_ros2_environment() -> None:
    """ROS2 Python/shared library 경로가 없으면 보강한 환경으로 앱을 한 번 재실행합니다."""
    if os.environ.get(ROS2_REEXEC_FLAG) == "1" or not ROS2_PREFIX.exists():
        return
    if not sys.argv or sys.argv[0] in ("-c", ""):
        return
    python_paths = [ROS2_LOCAL_PYTHON, ROS2_GLOBAL_PYTHON]
    lib_paths = [path for path in ROS2_LIB_DIRS if path.exists()]
    if not python_paths[0].exists() or not lib_paths:
        return

    env = dict(os.environ)
    env[ROS2_REEXEC_FLAG] = "1"
    env["PYTHONPATH"] = _prepend_env_paths(env.get("PYTHONPATH", ""), python_paths)
    env["LD_LIBRARY_PATH"] = _prepend_env_paths(env.get("LD_LIBRARY_PATH", ""), lib_paths)
    env["AMENT_PREFIX_PATH"] = _prepend_env_paths(env.get("AMENT_PREFIX_PATH", ""), [ROS2_PREFIX])
    env["CMAKE_PREFIX_PATH"] = _prepend_env_paths(env.get("CMAKE_PREFIX_PATH", ""), [ROS2_PREFIX])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _prepend_env_paths(current_value: str, paths: List[Path]) -> str:
    """환경 변수 path 목록 앞에 존재하는 경로를 중복 없이 추가합니다."""
    existing = [item for item in current_value.split(os.pathsep) if item]
    additions = [str(path) for path in paths if path.exists()]
    merged = additions + [item for item in existing if item not in additions]
    return os.pathsep.join(merged)


class LabelingApp(QMainWindow, Ui_MainWindow):
    """라벨링 앱의 메인 컨트롤러입니다."""

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle(APP_TITLE)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self.session: Optional[DatasetSession] = None
        self.label_file_path: Optional[Path] = None
        self.current_index = 0
        self.total_frames = 0
        self.timeline_start_sec = 0.0
        self.timeline_end_sec = 0.0
        self.namespace_config = load_yaml(DEFAULT_VIEW_CONFIG_FILE)
        self.view_config_path = DEFAULT_VIEW_CONFIG_FILE

        self.label_storage = LabelStorage(DEFAULT_LABEL_DATASET, self._log)
        self.labeling_logic = LabelingLogic(CLASS_COLORS)
        self.workspace_splitter: Optional[QSplitter] = None
        self.data_docks: List[DataDockWidget] = []
        self.data_view_splitter: Optional[QSplitter] = None
        self.floating_view_windows: Dict[int, QDialog] = {}
        self.dock_locations: Dict[int, tuple[QSplitter, int]] = {}

        self._setup_ui_components()
        self._connect_signals()
        self._update_ui_state()

    # ==========================================================================
    # UI setup
    # ==========================================================================

    def _setup_ui_components(self) -> None:
        self.timeline_slider = self._replace_timeline_slider()
        self.spinBox_viewCount.setValue(PREVIEW_DATA_VIEW_COUNT)
        self.comboBox_viewBackground.setCurrentText(DEFAULT_DATA_VIEW_BACKGROUND)
        self.doubleSpinBox_zoomSeconds.setValue(self._default_zoom_seconds())
        self._remove_unused_view_setting_rows()
        self._setup_view_settings_buttons()
        self._swap_log_and_class_view_locations()
        self._setup_workspace_splitter()
        self._setup_timeline_controls()
        self._setup_splitter_geometry()
        self._build_data_views(PREVIEW_DATA_VIEW_COUNT)

    def _remove_unused_view_setting_rows(self) -> None:
        """UI에서 제거된 데이터 뷰 설정 항목의 폼 행까지 정리합니다."""
        self.formLayout_viewSettings.setVerticalSpacing(VIEW_SETTINGS_ROW_SPACING)
        self.formLayout_viewSettings.setHorizontalSpacing(VIEW_SETTINGS_COLUMN_SPACING)
        for label_widget in (self.label_viewBackground, self.label_scaleMode):
            self._take_view_settings_row(label_widget)

    def _take_view_settings_row(self, label_widget) -> None:
        """폼 레이아웃에서 label_widget이 포함된 행을 제거합니다."""
        for widget in self._take_view_settings_row_widgets(label_widget):
            widget.deleteLater()

    def _take_view_settings_row_widgets(self, row_widget) -> List[QWidget]:
        """폼 레이아웃에서 row_widget이 포함된 행을 분리하고 위젯 목록을 반환합니다."""
        widgets = []
        row, _role = self.formLayout_viewSettings.getWidgetPosition(row_widget)
        if row < 0:
            return widgets
        row_items = self.formLayout_viewSettings.takeRow(row)
        for item in (row_items.labelItem, row_items.fieldItem):
            widget = self._detach_form_item_widget(item)
            if widget is not None:
                widgets.append(widget)
        return widgets

    def _detach_form_item_widget(self, item) -> Optional[QWidget]:
        """폼 행에서 분리된 위젯의 부모를 해제하고 반환합니다."""
        if item is None:
            return None
        widget = item.widget()
        if widget is None:
            return None
        widget.setParent(None)
        return widget

    def _setup_view_settings_buttons(self) -> None:
        """Data View Settings 하위 버튼 배치를 구성합니다."""
        self.pushButton_saveViewConfig.setText("Save Config")
        self._move_shift_buttons_to_view_settings()
        self._add_fit_all_button_to_view_settings()
        self._move_config_buttons_to_equal_row()
        self._normalize_view_settings_controls()

    def _move_shift_buttons_to_view_settings(self) -> None:
        """Left/Right 전체 shift 버튼을 Data View Settings로 이동합니다."""
        for button in (self.pushButton_left, self.pushButton_right):
            self.gridLayout_fileButtons.removeWidget(button)
        self.gridLayout_fileButtons.removeWidget(self.pushButton_importLabel)
        self.gridLayout_fileButtons.addWidget(self.pushButton_importLabel, 2, 0, 1, 2)
        row_widget = self._make_equal_button_row(
            [self.pushButton_left, self.pushButton_right],
            "ViewSettingsShiftButtons",
        )
        self.formLayout_viewSettings.addRow(self._make_view_settings_label("Shift"), row_widget)

    def _add_fit_all_button_to_view_settings(self) -> None:
        """모든 Data View의 Fit과 동일한 동작을 수행하는 버튼을 추가합니다."""
        self.pushButton_fitAllViews = QPushButton("Fit All", self.groupBox_viewSettings)
        self.pushButton_fitAllViews.setObjectName("FitAllDataViewsButton")
        self.pushButton_fitAllViews.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.formLayout_viewSettings.addRow(
            self._make_view_settings_label("Fit"),
            self.pushButton_fitAllViews,
        )

    def _move_config_buttons_to_equal_row(self) -> None:
        """Reload/Save Config 버튼을 동일한 비율의 한 줄 버튼으로 재배치합니다."""
        self._take_view_settings_row_widgets(self.pushButton_reloadViewConfig)
        row_widget = self._make_equal_button_row(
            [self.pushButton_reloadViewConfig, self.pushButton_saveViewConfig],
            "ViewSettingsConfigButtons",
        )
        self.formLayout_viewSettings.addRow(row_widget)

    def _make_equal_button_row(self, buttons: List[QPushButton], object_name: str) -> QWidget:
        """동일한 stretch 비율을 가지는 버튼 행을 생성합니다."""
        row_widget = QWidget(self.groupBox_viewSettings)
        row_widget.setObjectName(object_name)
        row_widget.setFixedHeight(VIEW_SETTINGS_CONTROL_HEIGHT)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(VIEW_SETTINGS_BUTTON_SPACING)
        for button in buttons:
            button.setParent(row_widget)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setFixedHeight(VIEW_SETTINGS_CONTROL_HEIGHT)
            row_layout.addWidget(button, 1)
        return row_widget

    def _make_view_settings_label(self, text: str) -> QLabel:
        """설정 폼의 라벨을 동일한 스타일/정렬 기준으로 생성합니다."""
        label = QLabel(text, self.groupBox_viewSettings)
        label.setObjectName("ViewSettingsFormLabel")
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(VIEW_SETTINGS_CONTROL_HEIGHT)
        return label

    def _normalize_view_settings_controls(self) -> None:
        """Data View Settings의 라벨/버튼 크기를 동일 기준으로 맞춥니다."""
        self._normalize_view_settings_label_column()
        self._normalize_view_settings_button_heights()

    def _normalize_view_settings_label_column(self) -> None:
        """Data View Settings의 왼쪽 라벨 열을 동일 폭/높이와 중앙 정렬로 맞춥니다."""
        label_widgets = self._view_settings_label_widgets()
        if not label_widgets:
            return
        width = max(widget.sizeHint().width() for widget in label_widgets)
        width = max(VIEW_SETTINGS_LABEL_MIN_WIDTH, min(width, VIEW_SETTINGS_LABEL_MAX_WIDTH))
        self.formLayout_viewSettings.setLabelAlignment(Qt.AlignCenter)
        for widget in label_widgets:
            widget.setObjectName("ViewSettingsFormLabel")
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width)
            widget.setFixedHeight(VIEW_SETTINGS_CONTROL_HEIGHT)
            widget.setSizePolicy(QSizePolicy.Fixed, widget.sizePolicy().verticalPolicy())
            if hasattr(widget, "setAlignment"):
                widget.setAlignment(Qt.AlignCenter)

    def _normalize_view_settings_button_heights(self) -> None:
        """Data View Settings 하위 버튼 높이를 라벨 박스와 동일하게 맞춥니다."""
        buttons = [
            self.pushButton_left,
            self.pushButton_right,
            self.pushButton_fitAllViews,
            self.pushButton_reloadViewConfig,
            self.pushButton_saveViewConfig,
        ]
        for button in buttons:
            button.setFixedHeight(VIEW_SETTINGS_CONTROL_HEIGHT)

    def _view_settings_label_widgets(self) -> List[QWidget]:
        """폼의 LabelRole에 배치된 위젯을 반환합니다."""
        widgets = []
        for row in range(self.formLayout_viewSettings.rowCount()):
            item = self.formLayout_viewSettings.itemAt(row, QFormLayout.LabelRole)
            if item is not None and item.widget() is not None:
                widgets.append(item.widget())
        return widgets

    def _swap_log_and_class_view_locations(self) -> None:
        """오른쪽 패널에서 Log View와 Class View의 위치를 교체합니다."""
        layout = self.verticalLayout_sidePanel
        layout.removeWidget(self.groupBox_logView)
        layout.removeWidget(self.groupBox_classView)
        layout.insertWidget(0, self.groupBox_logView)
        layout.insertWidget(2, self.groupBox_classView)
        self._set_side_panel_stretch()

    def _set_side_panel_stretch(self) -> None:
        """제거된 설정 공간이 Log/Class View로 더 배분되도록 stretch를 설정합니다."""
        stretch_by_widget = {
            self.groupBox_logView: 5,
            self.groupBox_viewSettings: 0,
            self.groupBox_classView: 5,
            self.fileButtonsFrame: 0,
        }
        for index in range(self.verticalLayout_sidePanel.count()):
            item = self.verticalLayout_sidePanel.itemAt(index)
            widget = item.widget()
            self.verticalLayout_sidePanel.setStretch(index, stretch_by_widget.get(widget, 0))

    def _setup_workspace_splitter(self) -> None:
        """데이터/메뉴 영역과 타임라인 영역 사이에 세로 크기 조절 바를 추가합니다."""
        if self.workspace_splitter is not None:
            return
        root_layout = self.centralwidget.layout()
        root_layout.removeWidget(self.splitter_main)
        root_layout.removeWidget(self.timelinePanel)

        self.workspace_splitter = QSplitter(Qt.Vertical, self.centralwidget)
        self.workspace_splitter.setObjectName("WorkspaceTimelineSplitter")
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self.splitter_main)
        self.workspace_splitter.addWidget(self.timelinePanel)
        root_layout.insertWidget(0, self.workspace_splitter, 1)

        self.timelinePanel.setMinimumHeight(TIMELINE_PANEL_MIN_HEIGHT)
        self.timelinePanel.setMaximumHeight(TIMELINE_PANEL_MAX_HEIGHT)
        self.timelinePanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def _setup_timeline_controls(self) -> None:
        """타임라인 버튼을 세 그룹별 너비와 공통 높이로 설정합니다."""
        for group_name, buttons in self._timeline_button_groups().items():
            width = int(TIMELINE_BUTTON_WIDTHS[group_name])
            for button in buttons:
                button.setFixedHeight(TIMELINE_BUTTON_HEIGHT)
                button.setFixedWidth(width)
                button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._set_timeline_button_layout_stretch()

    def _timeline_control_buttons(self) -> List:
        buttons = []
        for group_buttons in self._timeline_button_groups().values():
            buttons.extend(group_buttons)
        return buttons

    def _timeline_button_groups(self) -> Dict[str, List]:
        return {
            "boundary": [self.pushButton_prevBoundary, self.pushButton_nextBoundary],
            "frame": [self.pushButton_prevFrame, self.pushButton_nextFrame],
            "label": [self.pushButton_toggleLabeling],
        }

    def _set_timeline_button_layout_stretch(self) -> None:
        for index in range(self.horizontalLayout_controls.count()):
            item = self.horizontalLayout_controls.itemAt(index)
            widget = item.widget()
            self.horizontalLayout_controls.setStretch(index, 0)

    def _setup_splitter_geometry(self) -> None:
        """초기 화면에서 오른쪽 설정 패널이 과도하게 커지지 않도록 제한합니다."""
        self.sidePanel.setMinimumWidth(SIDE_PANEL_MIN_WIDTH)
        self.sidePanel.setMaximumWidth(SIDE_PANEL_MAX_WIDTH)
        data_width = max(DEFAULT_WINDOW_WIDTH - SIDE_PANEL_INITIAL_WIDTH - 40, 800)
        self.splitter_main.setSizes([data_width, SIDE_PANEL_INITIAL_WIDTH])
        self.splitter_main.setStretchFactor(0, 1)
        self.splitter_main.setStretchFactor(1, 0)
        if self.workspace_splitter is not None:
            top_height = max(DEFAULT_WINDOW_HEIGHT - TIMELINE_PANEL_INITIAL_HEIGHT - 80, 640)
            self.workspace_splitter.setSizes([top_height, TIMELINE_PANEL_INITIAL_HEIGHT])
            self.workspace_splitter.setStretchFactor(0, 1)
            self.workspace_splitter.setStretchFactor(1, 0)

    def _replace_timeline_slider(self) -> SegmentedSlider:
        original = self.horizontalSlider_timeline
        slider = SegmentedSlider(Qt.Horizontal, SLIDER_STYLE, self.timelinePanel)
        slider.setObjectName("SegmentedSlider")
        layout = original.parentWidget().layout()
        layout.replaceWidget(original, slider)
        original.hide()
        original.deleteLater()
        return slider

    def _build_data_views(self, count: int) -> None:
        self._clear_data_views()
        columns = max(1, int(math.ceil(math.sqrt(count))))
        rows = max(1, int(math.ceil(count / columns)))
        self.data_view_splitter = QSplitter(Qt.Vertical, self.scrollAreaWidgetContents_dataViews)
        self.data_view_splitter.setObjectName("DataViewGridSplitter")
        self.data_view_splitter.setChildrenCollapsible(False)
        self.gridLayout_dataViews.addWidget(self.data_view_splitter, 0, 0)

        for index in range(count):
            if index % columns == 0:
                row_splitter = self._make_data_view_row_splitter(index // columns)
                self.data_view_splitter.addWidget(row_splitter)
            dock = DataDockWidget(
                f"Data {index + 1}",
                self.assign_stream_to_view,
                index,
                self.seek_to_timestamp,
                self.toggle_data_view_popup,
                self,
            )
            self.data_docks.append(dock)
            row_splitter.addWidget(dock)
            row_splitter.setStretchFactor(row_splitter.count() - 1, 1)
        self._balance_data_view_splitters(rows, columns)
        self.apply_view_interaction_settings(log=False)
        if self.session is not None:
            self._configure_views_for_session()

    def _make_data_view_row_splitter(self, row_index: int) -> QSplitter:
        """데이터 뷰 한 행을 구성하는 horizontal splitter를 생성합니다."""
        splitter = QSplitter(Qt.Horizontal, self.data_view_splitter)
        splitter.setObjectName(f"DataViewRowSplitter_{row_index + 1}")
        splitter.setChildrenCollapsible(False)
        return splitter

    def _balance_data_view_splitters(self, rows: int, columns: int) -> None:
        """초기에는 균등한 격자처럼 보이도록 splitter 비율을 맞춥니다."""
        if self.data_view_splitter is None:
            return
        for row_index in range(self.data_view_splitter.count()):
            row_splitter = self.data_view_splitter.widget(row_index)
            if isinstance(row_splitter, QSplitter):
                row_splitter.setSizes([1 for _ in range(max(row_splitter.count(), 1))])
        self.data_view_splitter.setSizes([1 for _ in range(rows)])
        for row_index in range(rows):
            self.data_view_splitter.setStretchFactor(row_index, 1)

    def _clear_data_views(self) -> None:
        self._close_floating_data_views()
        for dock in self.data_docks:
            dock.setParent(None)
            dock.deleteLater()
        while self.gridLayout_dataViews.count():
            item = self.gridLayout_dataViews.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.data_docks.clear()
        self.data_view_splitter = None
        self.dock_locations.clear()

    def _close_floating_data_views(self) -> None:
        for dialog in list(self.floating_view_windows.values()):
            dialog.blockSignals(True)
            dialog.close()
            dialog.deleteLater()
        self.floating_view_windows.clear()

    def toggle_data_view_popup(self, view_index: int) -> None:
        """Data view를 floating popup으로 분리하거나 원래 splitter 위치로 복귀시킵니다."""
        if view_index in self.floating_view_windows:
            self._dock_data_view(view_index, close_window=True)
            return
        self._pop_out_data_view(view_index)

    def _pop_out_data_view(self, view_index: int) -> None:
        if view_index >= len(self.data_docks):
            return
        dock = self.data_docks[view_index]
        parent_splitter = dock.parentWidget()
        if not isinstance(parent_splitter, QSplitter):
            self._log(f"Data {view_index + 1} cannot be popped out from this parent.", "WARN")
            return
        insert_index = parent_splitter.indexOf(dock)
        if insert_index < 0:
            self._log(f"Data {view_index + 1} splitter location was not found.", "WARN")
            return

        self.dock_locations[view_index] = (parent_splitter, insert_index)
        dialog = self._make_floating_data_view_window(view_index, dock)
        self.floating_view_windows[view_index] = dialog
        dock.set_popped_out(True)
        dialog.show()

    def _make_floating_data_view_window(self, view_index: int, dock: DataDockWidget) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Data {view_index + 1}")
        dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        dialog.resize(max(dock.width(), 720), max(dock.height(), 520))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(dock)
        dialog.finished.connect(lambda _result, idx=view_index: self._dock_data_view(idx, close_window=False))
        return dialog

    def _dock_data_view(self, view_index: int, close_window: bool) -> None:
        dock = self.data_docks[view_index] if view_index < len(self.data_docks) else None
        dialog = self.floating_view_windows.pop(view_index, None)
        location = self.dock_locations.pop(view_index, None)
        if dock is None or location is None:
            return

        parent_splitter, insert_index = location
        dock.setParent(None)
        parent_splitter.insertWidget(min(insert_index, parent_splitter.count()), dock)
        parent_splitter.setStretchFactor(parent_splitter.indexOf(dock), 1)
        dock.set_popped_out(False)
        dock.show()

        if dialog is not None:
            if close_window and dialog.isVisible():
                dialog.blockSignals(True)
                dialog.close()
            dialog.deleteLater()

    def _connect_signals(self) -> None:
        self.pushButton_import.clicked.connect(self.import_data)
        self.pushButton_importLabel.clicked.connect(self.import_label_data)
        self.pushButton_export.clicked.connect(self.export_data)
        self.pushButton_previous.clicked.connect(self.go_to_previous_file)
        self.pushButton_next.clicked.connect(self.go_to_next_file)
        self.pushButton_left.clicked.connect(lambda: self._switch_arm_view("left"))
        self.pushButton_right.clicked.connect(lambda: self._switch_arm_view("right"))
        self.pushButton_toggleLabeling.clicked.connect(self.toggle_labeling)
        self.pushButton_prevFrame.clicked.connect(self.prev_frame)
        self.pushButton_nextFrame.clicked.connect(self.next_frame)
        self.pushButton_prevBoundary.clicked.connect(lambda: self._move_to_boundary(-1))
        self.pushButton_nextBoundary.clicked.connect(lambda: self._move_to_boundary(1))
        self.pushButton_reloadViewConfig.clicked.connect(self.reload_view_config)
        self.pushButton_saveViewConfig.clicked.connect(self.save_current_view_config)
        self.pushButton_fitAllViews.clicked.connect(self.fit_all_data_views)
        self.timeline_slider.valueChanged.connect(self.slider_changed)
        self.listWidget_classView.itemDoubleClicked.connect(self.edit_segment_from_list)
        self.spinBox_viewCount.valueChanged.connect(self._build_data_views)
        self.comboBox_plotMode.currentTextChanged.connect(self.apply_view_interaction_settings)
        self.doubleSpinBox_zoomSeconds.valueChanged.connect(self.apply_view_interaction_settings)
        self.checkBox_mouseZoom.stateChanged.connect(self.apply_view_interaction_settings)
        self.checkBox_tooltip.stateChanged.connect(self.apply_view_interaction_settings)

    # ==========================================================================
    # File import/export
    # ==========================================================================

    def import_data(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in SUPPORTED_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "",
            f"Supported Files ({patterns});;HDF5 Files (*.h5 *.hdf5);;MCAP Files (*.mcap)",
        )
        if file_path:
            self._load_file(Path(file_path))

    def _load_file(self, file_path: Path, file_list: Optional[List[Path]] = None) -> None:
        try:
            self._log(f"Loading '{file_path.name}'...")
            file_list = file_list or discover_supported_files(file_path)
            loader = DatasetLoader(
                progress_callback=self._on_load_progress,
                max_image_frames=PREVIEW_MAX_IMAGE_FRAMES,
                max_numeric_samples=PREVIEW_MAX_NUMERIC_SAMPLES,
                hdf5_mapping_path=DEFAULT_HDF5_MAPPING_FILE,
                full_data=True,
            )
            self.session = loader.load_exact(file_path, file_list) if file_list else loader.load(file_path)
            self.label_file_path = None
            self.total_frames = self._infer_total_frames()
            self._sync_timeline_bounds_from_session()
            self.current_index = 0
            self.labeling_logic.reset()
            self._load_existing_labels()
            self._configure_views_for_session()
            self._reset_timeline()
            self._update_class_list()
            self._update_slider_segments()
            self._update_ui_state()
            self._log(f"Loaded {len(self.session.streams)} streams. Frames: {self.total_frames}")
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Failed to load file:\n{exc}")
            self._log(f"Failed to load file: {exc}", "ERROR")
            self.session = None
            self.label_file_path = None
            self.total_frames = 0
            self.timeline_start_sec = 0.0
            self.timeline_end_sec = 0.0
            self._update_ui_state()

    def _load_existing_labels(self) -> None:
        if self.session is None or self.session.source_kind != "hdf5":
            return
        self.labeling_logic.segments = self.label_storage.load_from_hdf5(self.session.file_path)
        self._log(f"Loaded {len(self.labeling_logic.segments)} existing labels.")

    def import_label_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Label File",
            "",
            "HDF5 Label Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not file_path:
            return
        try:
            result = self.label_storage.import_labels(Path(file_path))
            self._apply_label_import_result(Path(file_path), result)
        except Exception as exc:
            QMessageBox.critical(self, "Label Import Error", f"Failed to import labels:\n{exc}")
            self._log(f"Failed to import labels: {exc}", "ERROR")

    def _apply_label_import_result(self, file_path: Path, result: LabelImportResult) -> None:
        """원본 데이터 세션은 보존하고 라벨 segment/timeline만 갱신합니다."""
        self.label_file_path = file_path
        self.labeling_logic.reset()
        self.labeling_logic.segments = result.segments
        if self.session is None:
            self.total_frames = max(result.total_frames, self._max_label_frame() + 1, 1)
            self._sync_timeline_bounds_from_label(result)
            self.label_filePath.setText(f"Label: {file_path.name}")
        else:
            self._warn_if_label_length_mismatch(result.total_frames)
        self.current_index = min(self.current_index, max(self.total_frames - 1, 0))
        self._reset_timeline()
        self._update_class_list()
        self._update_slider_segments()
        self._update_ui_state()
        self._log(
            f"Imported labels: {file_path.name} "
            f"segments={len(result.segments)}, frames={result.total_frames}"
        )

    def _warn_if_label_length_mismatch(self, label_frames: int) -> None:
        if label_frames == self.total_frames:
            return
        self._log(
            "Imported label frame count differs from current data. "
            f"data={self.total_frames}, labels={label_frames}. "
            "Current data timeline is preserved.",
            "WARN",
        )

    def export_data(self) -> None:
        source_path = self.session.file_path if self.session is not None else self.label_file_path
        if source_path is None:
            return
        base_name = source_path.stem
        export_dir = source_path.parent / EXPORT_DIR_NAME
        output_path = export_dir / f"{base_name}_labeled.hdf5"
        try:
            self.label_storage.export(
                source_path,
                output_path,
                self.labeling_logic.segments,
                self.total_frames,
                (self.timeline_start_sec, self.timeline_end_sec),
            )
            self._log(f"Exported labels: {output_path}")
        except Exception as exc:
            self._log(f"Export failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_load_progress(self, percent: int, text: str) -> None:
        self.statusbar.showMessage(f"{percent:3d}% {text}")
        QApplication.processEvents()

    # ==========================================================================
    # Dynamic data views
    # ==========================================================================

    def _configure_views_for_session(self) -> None:
        if self.session is None:
            return
        names = self.session.stream_names
        defaults = self._default_stream_order(names)
        namespace_groups, namespace_labels = self._stream_namespace_groups(names)
        for dock in self.data_docks:
            dock.set_namespace_groups(namespace_groups, namespace_labels)
            dock.view.show_placeholder(f"Data {dock.index + 1}")
        self.apply_view_interaction_settings(log=False)
        for index, name in enumerate(defaults[:len(self.data_docks)]):
            self.assign_stream_to_view(index, name)
            self.data_docks[index].select_stream(name)
        self.label_filePath.setText(f"File: {self.session.file_path.name}")

    def assign_stream_to_view(self, view_index: int, stream_name: str) -> None:
        if self.session is None or view_index >= len(self.data_docks):
            return
        stream = self.session.get_stream(stream_name)
        dock = self.data_docks[view_index]
        dock.view.set_stream(stream, self.session.start_time_sec)
        dock.select_stream(stream_name)
        dock.view.update_timestamp(self._current_timestamp())

    def apply_view_interaction_settings(self, *args, log: bool = True) -> None:
        background = DEFAULT_DATA_VIEW_BACKGROUND
        image_scale = "Keep Aspect"
        plot_mode = self.comboBox_plotMode.currentText()
        zoom_seconds = float(self.doubleSpinBox_zoomSeconds.value())
        zoom_enabled = self.checkBox_mouseZoom.isChecked()
        tooltip_enabled = self.checkBox_tooltip.isChecked()
        overlay_config = self._overlay_config()
        for dock in self.data_docks:
            dock.view.default_range_sec = zoom_seconds
            dock.view.set_data_view_background(background)
            dock.view.set_image_scale_mode(image_scale)
            dock.view.set_plot_navigation_mode(plot_mode)
            dock.view.set_interaction_options(
                zoom_enabled,
                tooltip_enabled,
                1.15,
                int(overlay_config.get("precision", 6)),
                overlay_config,
            )
        if log:
            self._log("Data view settings applied.")

    def reload_view_config(self) -> None:
        self.namespace_config = load_yaml(self.view_config_path)
        self._configure_views_for_session()
        self._log(f"View config loaded: {self.view_config_path}")

    def save_current_view_config(self) -> None:
        data = dict(self.namespace_config)
        data["current_data_views"] = self._current_view_config_rows()
        data["overlay"] = self._overlay_config()
        save_path = self.view_config_path
        if not save_path.exists():
            save_path = DEFAULT_VIEW_CONFIG_FILE
        with save_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.namespace_config = data
        self._log(f"Current view config saved: {save_path}")

    def fit_all_data_views(self) -> None:
        """모든 데이터 뷰에 개별 Fit 버튼과 동일한 동작을 적용합니다."""
        for dock in self.data_docks:
            dock.view.set_full_range()
        self._log("Applied Fit to all data views.")

    def _current_view_config_rows(self) -> List[Dict[str, object]]:
        rows = []
        for dock in self.data_docks:
            stream = dock.combo.currentText()
            if not stream:
                continue
            rows.append(
                {
                    "view": dock.index + 1,
                    "namespace": dock.current_namespace(),
                    "stream": stream,
                }
            )
        return rows

    def _default_stream_order(self, names: List[str]) -> List[str]:
        configured = self._configured_default_streams(names)
        if configured:
            return configured + [name for name in names if name not in configured]
        image_names = [name for name in names if self.session.streams[name].stream_type == "image"]
        numeric_names = [name for name in names if self.session.streams[name].stream_type == "timeseries"]
        text_names = [name for name in names if self.session.streams[name].stream_type == "text"]
        return sorted(image_names, key=self._image_priority) + numeric_names + text_names

    def _configured_default_streams(self, names: List[str]) -> List[str]:
        configured = self.namespace_config.get("default_streams", [])
        if not isinstance(configured, list):
            return []
        available = set(names)
        return [str(name) for name in configured if str(name) in available]

    def _image_priority(self, name: str) -> tuple:
        priority = ["/left/cam/color", "/exo/cam/color", "/right/cam/color", "object_positions"]
        for index, token in enumerate(priority):
            if name.startswith(token):
                return index, name
        return len(priority), self._namespace_for_stream_name(name), name

    def _stream_namespace_groups(self, names: List[str]) -> tuple[dict, dict]:
        namespace_order = self._namespace_order()
        namespace_labels = self._namespace_labels()
        groups = {namespace_id: [] for namespace_id in namespace_order}
        for name in names:
            namespace_id = self._namespace_for_stream_name(name)
            groups.setdefault(namespace_id, []).append(name)
            namespace_labels.setdefault(namespace_id, namespace_id.title())
        return groups, namespace_labels

    def _namespace_order(self) -> List[str]:
        configured = self.namespace_config.get("namespace_order", [])
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured]
        return ["left", "exo", "right", "recog", "object", "metadata", "general"]

    def _namespace_labels(self) -> Dict[str, str]:
        definitions = self._namespace_definitions()
        labels = {}
        for namespace_id in self._namespace_order():
            item = definitions.get(namespace_id, {})
            labels[namespace_id] = str(item.get("label", namespace_id.title()))
        return labels

    def _namespace_definitions(self) -> dict:
        sources = self.namespace_config.get("sources", {})
        if not isinstance(sources, dict):
            return {}
        source_key = self._namespace_source_key()
        source_config = sources.get(source_key) or sources.get("default") or {}
        namespaces = source_config.get("namespaces", {})
        return namespaces if isinstance(namespaces, dict) else {}

    def _namespace_source_key(self) -> str:
        if self.session is None:
            return "default"
        if self.session.source_kind in ("mcap", "ros2bag"):
            return "mcap"
        if self.session.source_kind == "hdf5":
            return "hdf5"
        return self.session.source_kind or "default"

    def _namespace_for_stream_name(self, name: str) -> str:
        if name.startswith("metadata/"):
            return "metadata"
        definitions = self._namespace_definitions()
        for namespace_id in self._namespace_order():
            if namespace_id == "general":
                continue
            item = definitions.get(namespace_id, {})
            prefixes = item.get("prefixes", []) if isinstance(item, dict) else []
            if any(name.startswith(str(prefix)) for prefix in prefixes):
                return namespace_id
        return "general"

    def _overlay_config(self) -> dict:
        overlay = self.namespace_config.get("overlay", {})
        return overlay if isinstance(overlay, dict) else {}

    # ==========================================================================
    # Timeline and labels
    # ==========================================================================

    def _reset_timeline(self) -> None:
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, max(self.total_frames - 1, 0))
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.slider_changed(0)

    def slider_changed(self, value: int) -> None:
        self.current_index = int(value)
        timestamp_sec = self._current_timestamp()
        for dock in self.data_docks:
            dock.view.update_timestamp(timestamp_sec)
        if self.labeling_logic.is_labeling:
            self._update_preview_segment()
        self._update_timestamp_label()
        self._update_slider_segments()

    def seek_to_timestamp(self, timestamp_sec: float) -> None:
        if self.total_frames <= 1:
            return
        duration = self._timeline_duration_sec()
        ratio = 0.0 if duration <= 0.0 else (timestamp_sec - self.timeline_start_sec) / duration
        index = int(round(max(0.0, min(1.0, ratio)) * (self.total_frames - 1)))
        self.timeline_slider.setValue(index)

    def toggle_labeling(self) -> None:
        if self.total_frames <= 0:
            return
        if not self.labeling_logic.is_labeling:
            self.labeling_logic.start_labeling(self.current_index)
            self.pushButton_toggleLabeling.setText("Stop")
            self.pushButton_toggleLabeling.setStyleSheet("background-color: lightblue;")
            self._log(f"Labeling started at frame {self.current_index}.")
            return
        self._finish_labeling_segment()

    def _finish_labeling_segment(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Enter Class ID(s)",
            f"Class ID(s), max {MAX_CLASS_IDS_PER_SEGMENT} (e.g., 1 or 1, 2):",
        )
        if ok and text.strip():
            try:
                class_ids = [int(value) for value in text.replace(",", " ").split()]
                if not class_ids:
                    raise ValueError
                if len(class_ids) > MAX_CLASS_IDS_PER_SEGMENT:
                    QMessageBox.warning(self, "Input Error", "Too many class IDs.")
                    return
                self.labeling_logic.stop_labeling(self.current_index, class_ids)
                self._log(f"Segment added: Class(es): {class_ids}")
                self._update_class_list()
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Enter valid integer Class IDs.")
                return
        self._clear_labeling_state()

    def _clear_labeling_state(self) -> None:
        self.labeling_logic.is_labeling = False
        self.labeling_logic.label_start_frame = None
        self.pushButton_toggleLabeling.setText("Start")
        self.pushButton_toggleLabeling.setStyleSheet("")
        self.timeline_slider.set_preview_segment(None)
        self._update_slider_segments()

    def edit_segment_from_list(self, item: QListWidgetItem) -> None:
        if self.total_frames <= 0:
            return
        index = self.listWidget_classView.row(item)
        segment = self.labeling_logic.segments[index]
        dialog = EditSegmentDialog(segment, self.total_frames, self)
        if dialog.exec_():
            new_data = dialog.get_data()
            if new_data:
                self.labeling_logic.edit_segment(index, new_data)
                self._update_class_list()
                self._update_slider_segments()

    def _update_class_list(self) -> None:
        self.listWidget_classView.clear()
        for start, end, class_ids in sorted(self.labeling_logic.segments):
            item_text = f"[{start} - {end}] Class: {', '.join(map(str, class_ids))}"
            item = QListWidgetItem(item_text)
            item.setIcon(self._class_icon(class_ids))
            self.listWidget_classView.addItem(item)

    def _update_slider_segments(self) -> None:
        colored_segments = []
        max_index = max(self.total_frames - 1, 0)
        for start, end, class_ids in self.labeling_logic.segments:
            draw_start = max(0, min(start, max_index))
            draw_end = max(0, min(end, max_index))
            if draw_start > draw_end:
                continue
            color = Qt.yellow if len(class_ids) > 1 else self.labeling_logic.get_class_color(class_ids[0])
            colored_segments.append((draw_start, draw_end, color))
        self.timeline_slider.set_segments(colored_segments)

    def _update_preview_segment(self) -> None:
        start = min(self.labeling_logic.label_start_frame, self.current_index)
        end = max(self.labeling_logic.label_start_frame, self.current_index)
        color = self.labeling_logic.get_class_color(-1, temp=True)
        self.timeline_slider.set_preview_segment((start, end, color))

    def _class_icon(self, class_ids: List[int]) -> QIcon:
        color = Qt.yellow if len(class_ids) > 1 else self.labeling_logic.get_class_color(class_ids[0])
        pixmap = QPixmap(20, 20)
        pixmap.fill(color)
        return QIcon(pixmap)

    # ==========================================================================
    # Navigation
    # ==========================================================================

    def prev_frame(self) -> None:
        self.timeline_slider.setValue(max(self.timeline_slider.minimum(), self.timeline_slider.value() - 1))

    def next_frame(self) -> None:
        self.timeline_slider.setValue(min(self.timeline_slider.maximum(), self.timeline_slider.value() + 1))

    def go_to_next_file(self) -> None:
        self._navigate_file(1)

    def go_to_previous_file(self) -> None:
        self._navigate_file(-1)

    def _navigate_file(self, direction: int) -> None:
        if self.session is None:
            return
        next_index = self.session.file_index + direction
        if 0 <= next_index < len(self.session.file_list):
            self._load_file(self.session.file_list[next_index], self.session.file_list)

    def _move_to_boundary(self, direction: int) -> None:
        if not self.labeling_logic.segments or self.total_frames <= 0:
            return
        regions = self._labeled_and_empty_regions()
        current = next((item for item in regions if item[0] <= self.current_index <= item[1]), None)
        if current is None:
            return
        target = self._boundary_target(direction, current, regions)
        if target is not None:
            self.timeline_slider.setValue(target)

    def _labeled_and_empty_regions(self) -> List[tuple]:
        regions = []
        last_end = -1
        for start, end, _ in sorted(self.labeling_logic.segments):
            if start > last_end + 1:
                regions.append((last_end + 1, start - 1, False))
            regions.append((start, end, True))
            last_end = end
        if last_end < self.total_frames - 1:
            regions.append((last_end + 1, self.total_frames - 1, False))
        return regions

    def _boundary_target(self, direction: int, current: tuple, regions: List[tuple]) -> Optional[int]:
        if direction < 0:
            if self.current_index == current[0] and current[2]:
                previous = next((item for item in reversed(regions) if item[2] and item[0] < current[0]), None)
                return previous[0] if previous else None
            return current[0]
        if self.current_index == current[1] and current[2]:
            next_region = next((item for item in regions if item[2] and item[0] > current[1]), None)
            return next_region[1] if next_region else None
        return current[1]

    def _switch_arm_view(self, target_arm: str) -> None:
        if self.session is None:
            return
        source_arm = "right" if target_arm == "left" else "left"
        available = set(self.session.stream_names)
        for dock in self.data_docks:
            current = dock.combo.currentText()
            next_name = self._replace_arm_prefix(current, source_arm, target_arm)
            if next_name in available:
                self.assign_stream_to_view(dock.index, next_name)

    def _replace_arm_prefix(self, name: str, source_arm: str, target_arm: str) -> str:
        replacements = (
            (f"/{source_arm}/", f"/{target_arm}/"),
            (f"{source_arm}/", f"{target_arm}/"),
            (f"{source_arm}_", f"{target_arm}_"),
        )
        for old, new in replacements:
            if name.startswith(old):
                return name.replace(old, new, 1)
        return name

    # ==========================================================================
    # State helpers
    # ==========================================================================

    def _infer_total_frames(self) -> int:
        if self.session is None:
            return 0
        lengths = [
            len(stream.timestamps)
            for stream in self.session.streams.values()
            if stream.source_type != "metadata" and len(stream.timestamps)
        ]
        if lengths:
            return max(lengths)
        if self.session.source_kind == "mcap":
            return TIMELINE_STEPS
        return 0

    def _sync_timeline_bounds_from_session(self) -> None:
        if self.session is None:
            return
        self.timeline_start_sec = float(self.session.start_time_sec)
        self.timeline_end_sec = float(self.session.end_time_sec)
        if self.timeline_end_sec < self.timeline_start_sec:
            self.timeline_end_sec = self.timeline_start_sec

    def _sync_timeline_bounds_from_label(self, result: LabelImportResult) -> None:
        if result.timestamp_bounds is not None:
            self.timeline_start_sec = float(result.timestamp_bounds[0])
            self.timeline_end_sec = float(result.timestamp_bounds[1])
            return
        self.timeline_start_sec = 0.0
        self.timeline_end_sec = float(max(self.total_frames - 1, 0))

    def _timeline_duration_sec(self) -> float:
        return max(self.timeline_end_sec - self.timeline_start_sec, 0.0)

    def _max_label_frame(self) -> int:
        if not self.labeling_logic.segments:
            return 0
        return max(end for _, end, _ in self.labeling_logic.segments)

    def _current_timestamp(self) -> float:
        if self.total_frames <= 1:
            return self.timeline_start_sec
        ratio = self.current_index / max(self.total_frames - 1, 1)
        return self.timeline_start_sec + self._timeline_duration_sec() * ratio

    def _default_zoom_seconds(self) -> float:
        timeline = self.namespace_config.get("timeline", {})
        if isinstance(timeline, dict) and "default_zoom_seconds" in timeline:
            return float(timeline["default_zoom_seconds"])
        return DEFAULT_ZOOM_SECONDS

    def _update_timestamp_label(self) -> None:
        if self.total_frames <= 0:
            self.label_timestamp.setText("Timestamp: 0 / 0 | Class: None")
            return
        rel_time = self._current_timestamp() - self.timeline_start_sec
        class_text = self.labeling_logic.get_class_at(self.current_index)
        self.label_timestamp.setText(
            f"Frame: {self.current_index} / {max(self.total_frames - 1, 0)} | "
            f"Time: {rel_time:.3f}s | Class: {class_text}"
        )

    def _update_ui_state(self) -> None:
        has_timeline = self.total_frames > 0
        has_session = self.session is not None
        self.timeline_slider.setEnabled(has_timeline)
        self.pushButton_export.setEnabled(has_timeline)
        self.pushButton_toggleLabeling.setEnabled(has_timeline)
        self.pushButton_prevFrame.setEnabled(has_timeline)
        self.pushButton_nextFrame.setEnabled(has_timeline)
        self.pushButton_prevBoundary.setEnabled(has_timeline)
        self.pushButton_nextBoundary.setEnabled(has_timeline)
        self.pushButton_left.setEnabled(has_session)
        self.pushButton_right.setEnabled(has_session)
        self.pushButton_fitAllViews.setEnabled(has_session)
        if not has_timeline:
            self.label_filePath.setText("File: No file loaded.")
            self.label_timestamp.setText("Timestamp: 0 / 0 | Class: None")
            self.pushButton_previous.setEnabled(False)
            self.pushButton_next.setEnabled(False)
            return
        if not has_session:
            self.pushButton_previous.setEnabled(False)
            self.pushButton_next.setEnabled(False)
            return
        self.pushButton_previous.setEnabled(self.session.file_index > 0)
        self.pushButton_next.setEnabled(self.session.file_index < len(self.session.file_list) - 1)

    def _log(self, message: str, level: str = "INFO") -> None:
        stamp = time.strftime("%H:%M:%S")
        self.textBrowser_logView.append(f"[{stamp}][{level}] {message}")

    # ==========================================================================
    # Qt events
    # ==========================================================================

    def keyPressEvent(self, event) -> None:
        if event.key() in TOGGLE_LABELING_KEYS:
            self.toggle_labeling()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self.labeling_logic.is_labeling:
            self._clear_labeling_state()
            self._log("Labeling canceled.")
            event.accept()
            return
        if event.key() == Qt.Key_Delete and self.listWidget_classView.hasFocus():
            self._delete_selected_segments()
            event.accept()
            return
        super().keyPressEvent(event)

    def _delete_selected_segments(self) -> None:
        items = self.listWidget_classView.selectedItems()
        rows = sorted([self.listWidget_classView.row(item) for item in items], reverse=True)
        for row in rows:
            self.labeling_logic.delete_segment(row)
        self._update_class_list()
        self._update_slider_segments()


if __name__ == "__main__":
    bootstrap_ros2_environment()

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    stylesheet = load_stylesheet(STYLE_FILE)
    if stylesheet:
        app.setStyleSheet(stylesheet)

    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
