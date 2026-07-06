# app_config.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# YAML-backed application configuration
# ==============================================================================

from pathlib import Path
from typing import Any, Dict

import yaml
from PyQt5.QtCore import Qt


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
APP_SETTINGS_FILE = CONFIG_DIR / "app_settings.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """YAML 파일을 dict로 읽고, 없거나 비정상인 경우 빈 dict를 반환합니다."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def path_from_config(value: str, fallback: Path) -> Path:
    """설정 파일의 상대 경로를 프로젝트 루트 기준 절대 경로로 변환합니다."""
    if not value:
        return fallback
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


APP_SETTINGS = load_yaml(APP_SETTINGS_FILE)

# ==============================================================================
# 핵심 파일 및 앱 기본값
# ==============================================================================

APP_TITLE = APP_SETTINGS.get("app_title", "MulTRecog Labeling Tool")
MAIN_UI_FILE = path_from_config(
    APP_SETTINGS.get("main_ui_file", "main_window.ui"),
    BASE_DIR / "main_window.ui",
)
STYLE_FILE = path_from_config(
    APP_SETTINGS.get("style_file", "configs/style_light.qss"),
    CONFIG_DIR / "style_light.qss",
)
DEFAULT_HDF5_MAPPING_FILE = path_from_config(
    APP_SETTINGS.get("default_hdf5_mapping_file", "configs/hdf5_mapping.yaml"),
    CONFIG_DIR / "hdf5_mapping.yaml",
)
DEFAULT_VIEW_CONFIG_FILE = path_from_config(
    APP_SETTINGS.get("default_view_config_file", "configs/view_config.yaml"),
    CONFIG_DIR / "view_config.yaml",
)
DEFAULT_LABEL_DATASET = str(APP_SETTINGS.get("default_label_dataset", "labels"))
EXPORT_DIR_NAME = str(APP_SETTINGS.get("export_dir_name", "label"))
DEFAULT_APP_THEME = APP_SETTINGS.get("default_app_theme", "Light")
DEFAULT_DATA_VIEW_BACKGROUND = APP_SETTINGS.get("default_data_view_background", "Dark")

SUPPORTED_EXTENSIONS = tuple(
    APP_SETTINGS.get("supported_extensions", [".mcap", ".h5", ".hdf5"])
)

WINDOW_SETTINGS = APP_SETTINGS.get("window", {}) or {}
DEFAULT_WINDOW_WIDTH = int(WINDOW_SETTINGS.get("default_width", 1920))
DEFAULT_WINDOW_HEIGHT = int(WINDOW_SETTINGS.get("default_height", 1080))

# ==============================================================================
# 데이터 뷰 및 타임라인 설정
# ==============================================================================

PREVIEW_SETTINGS = APP_SETTINGS.get("preview", {}) or {}
PREVIEW_DATA_VIEW_COUNT = int(PREVIEW_SETTINGS.get("data_view_count", 9))
PREVIEW_IMAGE_TOPIC_LIMIT = int(PREVIEW_SETTINGS.get("image_topic_limit", 0))
PREVIEW_NUMERIC_TOPIC_LIMIT = int(PREVIEW_SETTINGS.get("numeric_topic_limit", 0))
PREVIEW_MAX_IMAGE_FRAMES = int(PREVIEW_SETTINGS.get("max_image_frames", 0))
PREVIEW_MAX_NUMERIC_SAMPLES = int(PREVIEW_SETTINGS.get("max_numeric_samples", 0))

TIMELINE_SETTINGS = APP_SETTINGS.get("timeline", {}) or {}
TIMELINE_STEPS = int(TIMELINE_SETTINGS.get("steps", 10000))
DEFAULT_ZOOM_SECONDS = float(TIMELINE_SETTINGS.get("default_zoom_seconds", 1.0))

# reference 로더가 사용하는 호환 alias입니다.
IMAGE_TOPIC_LIMIT = PREVIEW_IMAGE_TOPIC_LIMIT
NUMERIC_TOPIC_LIMIT = PREVIEW_NUMERIC_TOPIC_LIMIT
MAX_IMAGE_FRAMES = PREVIEW_MAX_IMAGE_FRAMES
MAX_NUMERIC_SAMPLES = PREVIEW_MAX_NUMERIC_SAMPLES

# ==============================================================================
# 라벨링 및 스타일 설정
# ==============================================================================

LABELING_SETTINGS = APP_SETTINGS.get("labeling", {}) or {}
MAX_CLASS_IDS_PER_SEGMENT = int(LABELING_SETTINGS.get("max_class_ids_per_segment", 2))
TOGGLE_LABELING_KEYS = [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space]

SLIDER_STYLE = {
    "GROOVE_HEIGHT": 8,
    "BORDER_WIDTH": 1,
    "GROOVE_COLOR": "#D8D4CE",
    "GROOVE_BORDER_COLOR": "#938C84",
    "HANDLE_COLOR": "#4A4640",
    "HANDLE_BORDER_COLOR": "#302D29",
    "HANDLE_WIDTH": 18,
    "HANDLE_HEIGHT": 18,
}

CLASS_COLORS = [
    Qt.red,
    Qt.green,
    Qt.blue,
    Qt.cyan,
    Qt.magenta,
    Qt.darkRed,
    Qt.darkGreen,
    Qt.darkBlue,
    Qt.darkYellow,
]
