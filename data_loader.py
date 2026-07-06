# data_loader.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Dataset import and rosbag preview loading
# ==============================================================================

import io
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from app_config import (
    DEFAULT_HDF5_MAPPING_FILE,
    DEFAULT_LABEL_DATASET,
    IMAGE_TOPIC_LIMIT,
    MAX_IMAGE_FRAMES,
    MAX_NUMERIC_SAMPLES,
    NUMERIC_TOPIC_LIMIT,
    SUPPORTED_EXTENSIONS,
)
from custom_msg_parser import LocalCustomMessageParser
from data_models import DataStream, DatasetSession, TopicInfo


ProgressCallback = Callable[[int, str], None]
ROSBAG_SCAN_PROGRESS_INTERVAL = 10000
HDF5_DATASET_PROGRESS_INTERVAL = 50
METADATA_STREAM_PREFIX = "metadata/"


def discover_supported_files(
    path: Path,
    extra_ignored_parts: Tuple[str, ...] = (),
) -> List[Path]:
    """Return sorted supported files from a file or folder path."""
    path = path.expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_generated_output(item, extra_ignored_parts)
    ]
    return _filter_highest_priority_format(files)


def _filter_highest_priority_format(files: List[Path]) -> List[Path]:
    """Keep only one file format group, preferring rosbag2 MCAP first."""
    if not files:
        return []
    sorted_files = sorted(files, key=lambda item: (_extension_priority(item), item.name))
    selected_group = _format_group(sorted_files[0])
    return [item for item in sorted_files if _format_group(item) == selected_group]


def _format_group(path: Path) -> str:
    """Return the logical format group used for workspace navigation."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return "hdf5"
    if suffix in (".jpg", ".jpeg", ".png"):
        return "image"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    return suffix.lstrip(".")


def _extension_priority(path: Path) -> int:
    priority = {
        ".mcap": 0,
        ".h5": 1,
        ".hdf5": 1,
        ".csv": 2,
        ".mp4": 3,
        ".jpg": 4,
        ".jpeg": 4,
        ".png": 4,
        ".txt": 5,
        ".yaml": 6,
        ".yml": 6,
    }
    return priority.get(path.suffix.lower(), 99)


def _is_generated_output(path: Path, extra_ignored_parts: Tuple[str, ...] = ()) -> bool:
    ignored_parts = {"analysis", "exports", "__pycache__", *extra_ignored_parts}
    return any(part in ignored_parts for part in path.parts)


def _add_metadata_stream(
    session: DatasetSession,
    name: str,
    title: str,
    lines: List[str],
) -> None:
    """Add a read-only text stream for session/file metadata."""
    stream_name = f"{METADATA_STREAM_PREFIX}{name}"
    timestamp = session.start_time_sec if session.start_time_sec else 0.0
    text = "\n".join([title, "=" * len(title), *lines]).rstrip() + "\n"
    session.streams[stream_name] = DataStream(
        stream_name,
        "text",
        np.array([timestamp], dtype=float),
        labels=[text],
        source_type="metadata",
    )


def _format_bytes(num_bytes: int) -> str:
    """Format byte counts for compact metadata text."""
    value = float(max(int(num_bytes), 0))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def _is_ros_image_message_type(message_type: str) -> bool:
    """Return whether a ROS message contains an RGB or depth image."""
    return "CompressedImage" in message_type or message_type.endswith("/Image")


def find_workspace_for_file(file_path: Path) -> Path:
    """Use the rosbag folder as workspace when metadata.yaml exists nearby."""
    if file_path.suffix.lower() == ".mcap" and (file_path.parent / "metadata.yaml").exists():
        return file_path.parent
    return file_path.parent


class DatasetLoader:
    """High-level loader dispatching by file extension."""

    def __init__(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        max_image_frames: int = MAX_IMAGE_FRAMES,
        max_numeric_samples: int = MAX_NUMERIC_SAMPLES,
        hdf5_mapping_path: Optional[Path] = DEFAULT_HDF5_MAPPING_FILE,
        full_data: bool = False,
    ):
        self.progress_callback = progress_callback or (lambda percent, text: None)
        self.max_image_frames = max_image_frames
        self.max_numeric_samples = max_numeric_samples
        self.hdf5_mapping_path = hdf5_mapping_path
        self.full_data = full_data

    def load(self, path: Path) -> DatasetSession:
        files = discover_supported_files(path)
        if not files:
            raise FileNotFoundError(f"No supported files found: {path}")

        selected = files[0]
        workspace = path.resolve() if path.is_dir() else find_workspace_for_file(selected)
        file_index = files.index(selected)
        session = self._load_file(selected, workspace, files, file_index)
        return session

    def load_exact(self, file_path: Path, file_list: List[Path]) -> DatasetSession:
        file_path = file_path.resolve()
        workspace = find_workspace_for_file(file_path)
        file_index = file_list.index(file_path) if file_path in file_list else 0
        return self._load_file(file_path, workspace, file_list, file_index)

    def _load_file(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        suffix = file_path.suffix.lower()
        if suffix == ".mcap":
            loader = RosbagPreviewLoader(
                self.progress_callback,
                self.max_image_frames,
                self.max_numeric_samples,
                self.full_data,
            )
            return loader.load(file_path, workspace, file_list, file_index)
        if suffix in (".jpg", ".jpeg", ".png"):
            return self._load_image(file_path, workspace, file_list, file_index)
        if suffix in (".txt", ".yaml", ".yml"):
            return self._load_text(file_path, workspace, file_list, file_index)
        if suffix == ".csv":
            return self._load_csv(file_path, workspace, file_list, file_index)
        if suffix in (".h5", ".hdf5"):
            return self._load_hdf5(file_path, workspace, file_list, file_index)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _base_session(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
        source_kind: str,
    ) -> DatasetSession:
        return DatasetSession(
            file_path=file_path,
            workspace_path=workspace,
            file_list=file_list,
            file_index=file_index,
            source_kind=source_kind,
        )

    def _load_image(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        self.progress_callback(10, f"Loading image: {file_path.name}")
        data = file_path.read_bytes()
        stream = DataStream(file_path.name, "image", np.array([0.0]), image_bytes=[data])
        session = self._base_session(file_path, workspace, file_list, file_index, "image")
        session.streams[stream.name] = stream
        session.end_time_sec = 1.0
        session.duration_sec = 1.0
        self.progress_callback(100, "Image import completed.")
        return session

    def _load_text(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        self.progress_callback(10, f"Loading text: {file_path.name}")
        text = file_path.read_text(encoding="utf-8", errors="replace")
        stream = DataStream(file_path.name, "text", np.array([0.0]), labels=[text])
        session = self._base_session(file_path, workspace, file_list, file_index, "text")
        session.streams[stream.name] = stream
        session.note_text = text
        session.end_time_sec = 1.0
        session.duration_sec = 1.0
        self.progress_callback(100, "Text import completed.")
        return session

    def _load_csv(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        self.progress_callback(10, f"Loading CSV: {file_path.name}")
        data = np.genfromtxt(file_path, delimiter=",", names=True)
        session = self._base_session(file_path, workspace, file_list, file_index, "csv")
        values, labels = self._csv_to_array(data)
        timestamps = np.arange(len(values), dtype=float)
        stream = DataStream(file_path.name, "timeseries", timestamps, values, labels=labels)
        session.streams[stream.name] = stream
        session.end_time_sec = float(max(len(values) - 1, 1))
        session.duration_sec = session.end_time_sec
        self.progress_callback(100, "CSV import completed.")
        return session

    def _csv_to_array(self, data) -> Tuple[np.ndarray, List[str]]:
        if data.dtype.names:
            labels = list(data.dtype.names)
            columns = [np.asarray(data[name], dtype=float) for name in labels]
            return np.column_stack(columns), labels
        array = np.atleast_2d(np.asarray(data, dtype=float))
        labels = [f"value_{idx}" for idx in range(array.shape[1])]
        return array, labels

    def _load_hdf5(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError("h5py is required to import HDF5 files.") from exc

        session = self._base_session(file_path, workspace, file_list, file_index, "hdf5")
        self.progress_callback(10, f"Loading HDF5: {file_path.name}")
        with h5py.File(file_path, "r") as h5_file:
            self._append_mapped_hdf5_streams(session, h5_file)
            datasets: List[Tuple[str, object]] = []
            h5_file.visititems(lambda name, obj: datasets.append((name, obj)))
            total = max(len(datasets), 1)
            for index, (name, obj) in enumerate(datasets, start=1):
                if not hasattr(obj, "shape"):
                    continue
                self._append_hdf5_dataset(session, name, obj)
                if self._should_emit_hdf5_dataset_progress(index, total):
                    percent = min(95, 35 + int(index * 55 / total))
                    self.progress_callback(percent, f"Scanning HDF5 dataset: {name}")
            self._finalize_session_time(session)
            self._append_hdf5_metadata_streams(session, h5_file, datasets)
        self.progress_callback(100, "HDF5 import completed.")
        return session

    def _append_hdf5_metadata_streams(
        self,
        session: DatasetSession,
        h5_file,
        objects: List[Tuple[str, object]],
    ) -> None:
        """Expose HDF5 file metadata as text streams in the metadata namespace."""
        self._append_hdf5_summary_metadata(session, h5_file, objects)
        self._append_hdf5_attributes_metadata(session, h5_file, objects)
        self._append_hdf5_training_statistics_metadata(session, h5_file)

    def _append_hdf5_summary_metadata(
        self,
        session: DatasetSession,
        h5_file,
        objects: List[Tuple[str, object]],
    ) -> None:
        group_count = sum(1 for _, obj in objects if not hasattr(obj, "shape"))
        dataset_count = sum(1 for _, obj in objects if hasattr(obj, "shape"))
        lines = [
            f"File: {session.file_path}",
            f"Workspace: {session.workspace_path}",
            f"Source kind: {session.source_kind}",
            f"Duration [s]: {session.duration_sec:.6f}",
            f"Root keys: {len(h5_file.keys())}",
            f"Groups: {group_count}",
            f"Datasets: {dataset_count}",
            "",
            "Root keys",
            "-" * 48,
        ]
        lines.extend(f"- {name}" for name in h5_file.keys())
        lines.extend(["", "Object summary", "-" * 48])
        lines.extend(self._hdf5_object_lines(objects, limit=120))
        _add_metadata_stream(session, "hdf5_summary", "HDF5 Summary", lines)

    def _append_hdf5_attributes_metadata(
        self,
        session: DatasetSession,
        h5_file,
        objects: List[Tuple[str, object]],
    ) -> None:
        lines = ["Root attributes", "-" * 48]
        lines.extend(self._hdf5_attribute_lines("/", h5_file.attrs))
        lines.extend(["", "Object attributes", "-" * 48])
        count = 0
        for name, obj in objects:
            if not hasattr(obj, "attrs") or not obj.attrs:
                continue
            lines.append(f"[/{name}]")
            lines.extend(self._hdf5_attribute_lines(name, obj.attrs))
            lines.append("")
            count += 1
            if count >= 120:
                lines.append("... truncated after 120 objects with attributes")
                break
        _add_metadata_stream(session, "hdf5_attributes", "HDF5 Attributes", lines)

    def _append_hdf5_training_statistics_metadata(self, session: DatasetSession, h5_file) -> None:
        if "__training_statistics__" not in h5_file:
            _add_metadata_stream(
                session,
                "training_statistics",
                "Training Statistics",
                ["No __training_statistics__ group found in this HDF5 file."],
            )
            return
        stats_group = h5_file["__training_statistics__"]
        lines = ["Group: /__training_statistics__", ""]
        lines.extend(["Attributes", "-" * 48])
        lines.extend(self._hdf5_attribute_lines("__training_statistics__", stats_group.attrs))
        for section in ("timeseries", "images"):
            lines.extend(["", section, "-" * 48])
            group = stats_group.get(section)
            if group is None:
                lines.append("(missing)")
                continue
            for name in group.keys():
                item = group[name]
                sample_count = item.attrs.get("sample_count", "")
                lines.append(f"- {name} | sample_count={sample_count}")
        _add_metadata_stream(session, "training_statistics", "Training Statistics", lines)

    def _hdf5_object_lines(
        self,
        objects: List[Tuple[str, object]],
        limit: int,
    ) -> List[str]:
        lines = []
        for index, (name, obj) in enumerate(objects):
            if index >= limit:
                lines.append(f"... truncated after {limit} objects")
                break
            if hasattr(obj, "shape"):
                lines.append(f"- /{name} | dataset shape={obj.shape} dtype={obj.dtype}")
            else:
                lines.append(f"- /{name} | group keys={len(obj.keys())}")
        return lines

    def _hdf5_attribute_lines(self, name: str, attrs) -> List[str]:
        if not attrs:
            return ["(none)"]
        return [
            f"- {key}: {self._format_hdf5_attr_value(value)}"
            for key, value in attrs.items()
        ]

    def _format_hdf5_attr_value(self, value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            if value.size > 12:
                return f"{value.tolist()[:12]} ... shape={value.shape}"
            return str(value.tolist())
        return str(value)

    def _append_hdf5_dataset(self, session: DatasetSession, name: str, dataset) -> None:
        if name == DEFAULT_LABEL_DATASET:
            return
        if name.startswith("__training_statistics__/"):
            return
        if name.endswith(("/timestamps", "/values", "/image_bytes", "/image_lengths")):
            return
        array = dataset[()]
        if self._looks_like_encoded_image_dataset(array, name):
            image_bytes = self._hdf5_bytes_list(dataset)
            timestamps = np.arange(len(image_bytes), dtype=float)
            self._add_image_stream(session, name, timestamps, image_bytes)
            return
        if np.issubdtype(array.dtype, np.number):
            values = np.asarray(array, dtype=float)
            if self._looks_like_image_array(values):
                self._append_image_array_dataset(session, name, values)
                return
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim == 2:
                timestamps = np.arange(values.shape[0], dtype=float)
                labels = [f"{name}_{idx}" for idx in range(values.shape[1])]
                session.streams[name] = DataStream(name, "timeseries", timestamps, values, labels=labels)

    def _should_emit_hdf5_dataset_progress(self, index: int, total: int) -> bool:
        return (
            index == 1
            or index == total
            or index % HDF5_DATASET_PROGRESS_INTERVAL == 0
        )

    def _append_mapped_hdf5_streams(self, session: DatasetSession, h5_file) -> None:
        config = self._load_hdf5_mapping()
        image_streams = config.get("image_streams", [])
        timeseries_streams = config.get("timeseries_streams", [])
        total = max(len(image_streams) + len(timeseries_streams), 1)
        progress_index = 0
        for item in image_streams:
            progress_index += 1
            self.progress_callback(
                10 + int(progress_index * 20 / total),
                f"Mapping HDF5 image stream: {item.get('topic', item.get('group', ''))}",
            )
            self._append_mapped_hdf5_image(session, h5_file, item)
        for item in timeseries_streams:
            progress_index += 1
            self.progress_callback(
                10 + int(progress_index * 20 / total),
                f"Mapping HDF5 timeseries stream: {item.get('topic', item.get('group', ''))}",
            )
            self._append_mapped_hdf5_timeseries(session, h5_file, item)
        self.progress_callback(32, "Scanning HDF5 grouped streams...")
        self._append_hdf5_group_streams(session, h5_file)

    def _load_hdf5_mapping(self) -> Dict:
        path = Path(self.hdf5_mapping_path) if self.hdf5_mapping_path else None
        if path is None or not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file) or {}

    def _append_mapped_hdf5_image(self, session: DatasetSession, h5_file, item: Dict) -> None:
        group = self._hdf5_group(h5_file, item.get("group", ""))
        if group is None:
            return
        bytes_name = item.get("image_bytes", "image_bytes")
        array_name = item.get("image_array", "images")
        timestamps = self._hdf5_timestamps(group, item.get("timestamps", "timestamps"))
        topic = item.get("topic") or self._group_topic(group, item.get("group", ""))
        if bytes_name in group:
            image_bytes = self._hdf5_bytes_list(group[bytes_name])
            self._add_image_stream(session, topic, timestamps, image_bytes)
        elif array_name in group:
            image_bytes = self._encode_image_array(group[array_name][()])
            self._add_image_stream(session, topic, timestamps, image_bytes)

    def _append_mapped_hdf5_timeseries(self, session: DatasetSession, h5_file, item: Dict) -> None:
        group = self._hdf5_group(h5_file, item.get("group", ""))
        values_name = item.get("values", "values")
        if group is None or values_name not in group:
            return
        values = np.asarray(group[values_name][()], dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        timestamps = self._hdf5_timestamps(group, item.get("timestamps", "timestamps"), len(values))
        topic = item.get("topic") or self._group_topic(group, item.get("group", ""))
        labels = self._hdf5_timeseries_labels(item, topic, values.shape[1])
        session.streams[topic] = DataStream(topic, "timeseries", timestamps, values, labels=labels)

    def _append_hdf5_group_streams(self, session: DatasetSession, h5_file) -> None:
        for group_name, obj in h5_file.items():
            if not hasattr(obj, "attrs"):
                continue
            stream_type = obj.attrs.get("stream_type", "")
            topic = self._group_topic(obj, group_name)
            if topic in session.streams:
                continue
            if stream_type == "image" and "image_bytes" in obj:
                timestamps = self._hdf5_timestamps(obj, "timestamps", len(obj["image_bytes"]))
                self._add_image_stream(session, topic, timestamps, self._hdf5_bytes_list(obj["image_bytes"]))
            elif stream_type == "timeseries" and "values" in obj:
                values = np.asarray(obj["values"][()], dtype=float)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                timestamps = self._hdf5_timestamps(obj, "timestamps", len(values))
                labels = self._hdf5_timeseries_labels({}, topic, values.shape[1])
                session.streams[topic] = DataStream(topic, "timeseries", timestamps, values, labels=labels)

    def _hdf5_group(self, h5_file, group_name: str):
        if not group_name or group_name not in h5_file:
            return None
        return h5_file[group_name]

    def _hdf5_timestamps(self, group, name: str, fallback_len: int = 0) -> np.ndarray:
        if name in group:
            return np.asarray(group[name][()], dtype=float)
        return np.arange(fallback_len, dtype=float)

    def _group_topic(self, group, group_name: str) -> str:
        attr_name = group.attrs.get("stream_name", "") if hasattr(group, "attrs") else ""
        if attr_name:
            return str(attr_name)
        return "/" + group_name.replace("__", "/").strip("/")

    def _hdf5_bytes_list(self, dataset) -> List[bytes]:
        items = dataset[()]
        image_bytes = []
        for item in items:
            if isinstance(item, (bytes, bytearray, np.bytes_)):
                image_bytes.append(item)
            elif isinstance(item, np.void):
                image_bytes.append(bytes(item))
            else:
                image_bytes.append(np.asarray(item, dtype=np.uint8).tobytes())
        return image_bytes

    def _hdf5_timeseries_labels(self, item: Dict, topic: str, width: int) -> List[str]:
        labels = [str(label) for label in (item.get("labels") or [])]
        if labels:
            return labels[:width]
        return [f"{topic}_{idx}" for idx in range(width)]

    def _add_image_stream(
        self,
        session: DatasetSession,
        topic: str,
        timestamps: np.ndarray,
        image_bytes: List[bytes],
    ) -> None:
        if not image_bytes:
            return
        if len(timestamps) != len(image_bytes):
            timestamps = np.arange(len(image_bytes), dtype=float)
        session.streams[topic] = DataStream(topic, "image", timestamps, image_bytes=image_bytes)

    def _looks_like_image_array(self, values: np.ndarray) -> bool:
        return values.ndim in (3, 4) and values.shape[-1] in (1, 3, 4)

    def _looks_like_encoded_image_dataset(self, array: np.ndarray, name: str) -> bool:
        """기존 라벨링 HDF5처럼 압축 이미지 bytes가 dataset에 직접 저장된 경우를 감지합니다."""
        if not hasattr(array, "dtype") or array.ndim != 1:
            return False
        dtype = array.dtype
        if dtype.kind not in ("O", "S", "V", "U"):
            return False
        lowered = name.lower()
        return any(token in lowered for token in ("image", "compressed", "camera", "rgb"))

    def _append_image_array_dataset(self, session: DatasetSession, name: str, values: np.ndarray) -> None:
        image_bytes = self._encode_image_array(values)
        timestamps = np.arange(len(image_bytes), dtype=float)
        session.streams[name] = DataStream(name, "image", timestamps, image_bytes=image_bytes)

    def _encode_image_array(self, values: np.ndarray) -> List[bytes]:
        array = np.asarray(values)
        if array.ndim == 3:
            array = array.reshape((1,) + array.shape)
        image_bytes = []
        for image in array:
            image_u8 = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
            if image_u8.shape[-1] == 3:
                image_u8 = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".png", image_u8)
            if ok:
                image_bytes.append(encoded.tobytes())
        return image_bytes

    def _finalize_session_time(self, session: DatasetSession) -> None:
        bounds = [
            (float(stream.timestamps[0]), float(stream.timestamps[-1]))
            for stream in session.streams.values()
            if len(stream.timestamps)
        ]
        if not bounds:
            session.start_time_sec = 0.0
            session.end_time_sec = 1.0
            session.duration_sec = 1.0
            return

        absolute_bounds = [item for item in bounds if item[1] > 1e6]
        usable_bounds = absolute_bounds or bounds
        session.start_time_sec = min(start for start, _ in usable_bounds)
        session.end_time_sec = max(end for _, end in usable_bounds)
        session.duration_sec = max(session.end_time_sec - session.start_time_sec, 1e-6)


class RosbagPreviewLoader:
    """Read preview streams from rosbag2 MCAP using installed ROS 2 messages."""

    def __init__(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        max_image_frames: int = MAX_IMAGE_FRAMES,
        max_numeric_samples: int = MAX_NUMERIC_SAMPLES,
        full_data: bool = False,
    ):
        self.progress_callback = progress_callback or (lambda percent, text: None)
        self.custom_parser = LocalCustomMessageParser()
        self.max_image_frames = max_image_frames
        self.max_numeric_samples = max_numeric_samples
        self.full_data = full_data

    def load(
        self,
        file_path: Path,
        workspace: Path,
        file_list: List[Path],
        file_index: int,
    ) -> DatasetSession:
        try:
            import rosbag2_py
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "MCAP 파일을 읽으려면 ROS2 rosbag2_py가 필요합니다. "
                "IDE에서 실행 중이면 main_app.py를 다시 시작하거나, 터미널에서 "
                "`source /opt/ros/humble/setup.bash` 후 preprocessing 환경으로 실행하세요. "
                f"원본 오류: {exc}"
            ) from exc

        bag_uri = self._bag_uri(file_path)
        metadata_dir = file_path.parent if (file_path.parent / "metadata.yaml").exists() else workspace
        metadata = self._read_metadata(metadata_dir)
        topic_info = self._topic_info_from_metadata(metadata)
        session = DatasetSession(
            file_path=file_path,
            workspace_path=workspace,
            file_list=file_list,
            file_index=file_index,
            topic_info=topic_info,
            source_kind="mcap",
        )
        self._apply_bag_time(session, metadata)

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        if not topic_info:
            topic_info = self._topic_info_from_reader(topic_types)
            session.topic_info = topic_info
        selected = self._select_topics(topic_info, topic_types)
        self._read_selected_messages(reader, session, selected, topic_types)
        self._finalize_streams(session)
        self._append_rosbag_metadata_streams(session, metadata, selected, topic_types)
        return session

    def _bag_uri(self, file_path: Path) -> Path:
        if (file_path.parent / "metadata.yaml").exists():
            return file_path.parent
        return file_path

    def _read_metadata(self, workspace: Path) -> Dict:
        metadata_path = workspace / "metadata.yaml"
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj) or {}

    def _topic_info_from_metadata(self, metadata: Dict) -> List[TopicInfo]:
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        topics = []
        for item in bag_info.get("topics_with_message_count", []):
            meta = item.get("topic_metadata", {})
            topics.append(
                TopicInfo(
                    name=meta.get("name", ""),
                    message_type=meta.get("type", ""),
                    message_count=int(item.get("message_count", 0)),
                    serialization_format=meta.get("serialization_format", ""),
                )
            )
        return sorted(topics, key=lambda topic: topic.name)

    def _topic_info_from_reader(self, topic_types: Dict[str, str]) -> List[TopicInfo]:
        topics = [
            TopicInfo(name=name, message_type=message_type, message_count=0)
            for name, message_type in topic_types.items()
        ]
        return sorted(topics, key=lambda topic: topic.name)

    def _apply_bag_time(self, session: DatasetSession, metadata: Dict) -> None:
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        start_ns = bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
        duration_ns = bag_info.get("duration", {}).get("nanoseconds", 0)
        session.message_count = int(bag_info.get("message_count", 0))
        session.start_time_sec = float(start_ns) / 1e9
        session.duration_sec = max(float(duration_ns) / 1e9, 1e-6)
        session.end_time_sec = session.start_time_sec + session.duration_sec

    def _append_rosbag_metadata_streams(
        self,
        session: DatasetSession,
        metadata: Dict,
        selected_topics: Iterable[str],
        topic_types: Dict[str, str],
    ) -> None:
        selected = set(selected_topics)
        self._append_rosbag_summary_metadata(session, metadata, selected)
        self._append_rosbag_topic_metadata(session, selected, topic_types)
        self._append_rosbag_raw_metadata(session, metadata)

    def _append_rosbag_summary_metadata(
        self,
        session: DatasetSession,
        metadata: Dict,
        selected_topics: set,
    ) -> None:
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        lines = [
            f"File: {session.file_path}",
            f"Workspace: {session.workspace_path}",
            f"Source kind: {session.source_kind}",
            f"Duration [s]: {session.duration_sec:.6f}",
            f"Message count: {session.message_count}",
            f"Topic count: {len(session.topic_info)}",
            f"Preview stream count: {len([s for s in session.streams.values() if s.source_type != 'metadata'])}",
            f"Selected preview topics: {len(selected_topics)}",
            "",
            "Storage",
            "-" * 48,
            f"Version: {bag_info.get('version', '')}",
            f"Storage identifier: {bag_info.get('storage_identifier', '')}",
            f"Relative file paths: {bag_info.get('relative_file_paths', [])}",
            "",
            "Selected preview topics",
            "-" * 48,
        ]
        lines.extend(f"- {name}" for name in sorted(selected_topics))
        _add_metadata_stream(session, "rosbag_summary", "ROS2 Bag Summary", lines)

    def _append_rosbag_topic_metadata(
        self,
        session: DatasetSession,
        selected_topics: set,
        topic_types: Dict[str, str],
    ) -> None:
        lines = [
            "Topic | Type | Messages | Data size | Preview",
            "-" * 96,
        ]
        for topic in session.topic_info:
            preview = "yes" if topic.name in selected_topics else "no"
            message_type = topic.message_type or topic_types.get(topic.name, "")
            lines.append(
                f"{topic.name} | {message_type} | "
                f"{topic.message_count} | {_format_bytes(topic.data_size_bytes)} | {preview}"
            )
        _add_metadata_stream(session, "ros2_topics", "ROS2 Topic Metadata", lines)

    def _append_rosbag_raw_metadata(self, session: DatasetSession, metadata: Dict) -> None:
        if metadata:
            raw_text = yaml.safe_dump(
                metadata,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ).splitlines()
        else:
            raw_text = ["metadata.yaml was not found or could not be parsed."]
        _add_metadata_stream(session, "raw_rosbag_metadata", "Raw metadata.yaml", raw_text)

    def _select_topics(self, topic_info: List[TopicInfo], topic_types: Dict[str, str]) -> List[str]:
        image_topics = self._rank_image_topics(topic_info)
        numeric_topics = self._rank_numeric_topics(topic_info, topic_types)
        selected = (
            self._take_by_limit(image_topics, IMAGE_TOPIC_LIMIT)
            + self._take_by_limit(numeric_topics, NUMERIC_TOPIC_LIMIT)
        )
        if self.full_data:
            selected = image_topics + numeric_topics
        return [topic for topic in selected if topic in topic_types]

    @staticmethod
    def _take_by_limit(topics: List[str], limit: int) -> List[str]:
        if limit <= 0:
            return topics
        return topics[:limit]

    def _rank_image_topics(self, topic_info: List[TopicInfo]) -> List[str]:
        topics = [
            topic.name
            for topic in topic_info
            if _is_ros_image_message_type(topic.message_type)
        ]
        priority = ["/left/cam/color", "/right/cam/color", "/exo/cam/color"]

        def key(name: str) -> Tuple[int, str]:
            for idx, prefix in enumerate(priority):
                if name.startswith(prefix):
                    return idx, name
            return len(priority), name

        return sorted(topics, key=key)

    def _rank_numeric_topics(self, topic_info: List[TopicInfo], topic_types: Dict[str, str]) -> List[str]:
        allowed = (
            "WrenchStamped",
            "PoseStamped",
            "JointState",
            "GripperStates",
            "MasterStates",
            "TaskRecognition",
            "CameraInfo",
            "std_msgs/msg/Bool",
        )
        topics = [topic.name for topic in topic_info if any(kind in topic.message_type for kind in allowed)]
        priority = (
            "ft/states",
            "gripper/states",
            "gripper/command",
            "master/states",
            "recog/states",
            "current/base_eef_pose",
            "current/world_eef_pose",
            "joint_position",
        )

        def key(name: str) -> Tuple[int, str]:
            for idx, token in enumerate(priority):
                if token in name:
                    return idx, name
            return len(priority), name

        return sorted([topic for topic in topics if topic in topic_types], key=key)

    def _read_selected_messages(
        self,
        reader,
        session: DatasetSession,
        selected: Iterable[str],
        topic_types: Dict[str, str],
    ) -> None:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        selected = set(selected)
        counters = {name: 0 for name in selected}
        buffers = {name: {"t": [], "v": [], "img": []} for name in selected}
        topic_byte_counts = {name: 0 for name in topic_types}
        topic_message_counts = {name: 0 for name in topic_types}
        strides = self._make_strides(session.topic_info, selected)
        total = max(session.message_count, 1)
        read_count = 0
        self.progress_callback(5, "Reading selected rosbag topics...")

        while reader.has_next():
            topic, raw_data, timestamp_ns = reader.read_next()
            read_count += 1
            topic_byte_counts[topic] = topic_byte_counts.get(topic, 0) + len(raw_data)
            topic_message_counts[topic] = topic_message_counts.get(topic, 0) + 1
            if topic in selected:
                counters[topic] += 1
                if counters[topic] % strides.get(topic, 1) == 0:
                    self._append_message(
                        buffers[topic],
                        topic_types[topic],
                        raw_data,
                        timestamp_ns,
                        deserialize_message,
                        get_message,
                    )
            if read_count % ROSBAG_SCAN_PROGRESS_INTERVAL == 0:
                percent = min(95, int(read_count * 90 / total) + 5)
                self.progress_callback(percent, f"Scanned {read_count:,} rosbag messages...")

        count_map = {topic.name: topic.message_count for topic in session.topic_info}
        for name, buffer in buffers.items():
            stream = self._buffer_to_stream(
                name,
                topic_types[name],
                buffer,
                count_map.get(name, 0),
                strides.get(name, 1),
            )
            if stream is not None:
                session.streams[name] = stream
        self._apply_topic_scan_stats(
            session,
            topic_types,
            topic_byte_counts,
            topic_message_counts,
        )
        self._append_topic_summary_streams(session, topic_types)
        self.progress_callback(98, "Preview stream assembly completed.")

    def _apply_topic_scan_stats(
        self,
        session: DatasetSession,
        topic_types: Dict[str, str],
        topic_byte_counts: Dict[str, int],
        topic_message_counts: Dict[str, int],
    ) -> None:
        topic_map = {topic.name: topic for topic in session.topic_info}
        for name, message_type in topic_types.items():
            topic = topic_map.get(name)
            if topic is None:
                topic = TopicInfo(name=name, message_type=message_type, message_count=0)
                session.topic_info.append(topic)
            topic.data_size_bytes = int(topic_byte_counts.get(name, 0))
            if topic.message_count <= 0:
                topic.message_count = int(topic_message_counts.get(name, 0))
        session.topic_info = sorted(session.topic_info, key=lambda item: item.name)

    def _append_topic_summary_streams(
        self,
        session: DatasetSession,
        topic_types: Dict[str, str],
    ) -> None:
        """Represent undecoded topics as text streams so every topic is visible."""
        for topic in session.topic_info:
            if topic.name in session.streams:
                continue
            message_type = topic.message_type or topic_types.get(topic.name, "")
            lines = [
                f"Topic: {topic.name}",
                f"Message type: {message_type}",
                f"Message count: {topic.message_count}",
                f"Data size: {_format_bytes(topic.data_size_bytes)}",
                f"Serialization: {topic.serialization_format}",
                "",
                "This topic is available as metadata because no numeric/image parser is configured.",
            ]
            session.streams[topic.name] = DataStream(
                topic.name,
                "text",
                np.array([session.start_time_sec], dtype=float),
                labels=["\n".join(lines) + "\n"],
                source_type=message_type,
                original_message_count=topic.message_count,
                sample_stride=1,
            )

    def _make_strides(self, topic_info: List[TopicInfo], selected: Iterable[str]) -> Dict[str, int]:
        count_map = {topic.name: max(topic.message_count, 1) for topic in topic_info}
        type_map = {topic.name: topic.message_type for topic in topic_info}
        strides = {}
        for name in selected:
            is_image = _is_ros_image_message_type(type_map.get(name, ""))
            limit = self.max_image_frames if is_image else self.max_numeric_samples
            if limit <= 0:
                strides[name] = 1
            else:
                strides[name] = max(1, int(np.ceil(count_map.get(name, 1) / limit)))
        return strides

    def _append_message(
        self,
        buffer: Dict[str, List],
        message_type: str,
        raw_data: bytes,
        timestamp_ns: int,
        deserialize_message,
        get_message,
    ) -> None:
        try:
            msg_class = get_message(message_type)
            message = deserialize_message(raw_data, msg_class)
            timestamp_sec = float(timestamp_ns) / 1e9
            if "CompressedImage" in message_type:
                buffer["t"].append(timestamp_sec)
                buffer["img"].append(bytes(message.data))
                return
            if message_type.endswith("/Image"):
                png_bytes = self._raw_image_png_bytes(message)
                if png_bytes is not None:
                    buffer["t"].append(timestamp_sec)
                    buffer["img"].append(png_bytes)
                return
            values = self._message_to_values(message_type, message)
            if values:
                buffer["t"].append(timestamp_sec)
                buffer["v"].append(values)
        except ValueError:
            raise
        except Exception:
            values = self.custom_parser.decode_values(message_type, raw_data)
            if values:
                buffer["t"].append(float(timestamp_ns) / 1e9)
                buffer["v"].append(values)

    def _raw_image_png_bytes(self, message) -> Optional[bytes]:
        """Encode supported raw ROS images as lossless PNG bytes."""
        encoding = str(getattr(message, "encoding", "")).strip().lower()
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, step)
        if encoding in ("16uc1", "mono16"):
            byte_order = ">u2" if bool(message.is_bigendian) else "<u2"
            image = rows[:, : width * 2].copy().view(byte_order).reshape(height, width)
            image = image.astype(np.uint16, copy=False)
        elif encoding in ("mono8", "8uc1"):
            image = rows[:, :width].copy()
        elif encoding in ("rgb8", "bgr8"):
            image = rows[:, : width * 3].copy().reshape(height, width, 3)
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif encoding == "32fc1":
            # A float depth stream requires an explicit quantization contract.
            return None
        else:
            raise ValueError(f"Unsupported ROS Image encoding for PNG conversion: {encoding}")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError(f"Failed to encode ROS Image as PNG: {encoding}")
        return encoded.tobytes()

    def _message_to_values(self, message_type: str, message) -> List[float]:
        if "WrenchStamped" in message_type:
            return [
                message.wrench.force.x,
                message.wrench.force.y,
                message.wrench.force.z,
                message.wrench.torque.x,
                message.wrench.torque.y,
                message.wrench.torque.z,
            ]
        if "PoseStamped" in message_type:
            pose = message.pose
            return [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        if "JointState" in message_type:
            values = list(message.position[:7])
            values.extend(list(message.velocity[:7]))
            values.extend(list(message.effort[:7]))
            return [float(value) for value in values]
        if "GripperStates" in message_type:
            return [
                float(message.status),
                float(message.object_status),
                float(message.position),
                float(message.current),
            ]
        if "MasterStates" in message_type:
            return [
                float(message.raw_position.x),
                float(message.raw_position.y),
                float(message.raw_position.z),
                float(message.raw_rpy.x),
                float(message.raw_rpy.y),
                float(message.raw_rpy.z),
                float(message.delta_pose.position.x),
                float(message.delta_pose.position.y),
                float(message.delta_pose.position.z),
                float(message.delta_pose.orientation.x),
                float(message.delta_pose.orientation.y),
                float(message.delta_pose.orientation.z),
                float(message.delta_pose.orientation.w),
                float(bool(message.is_indexing)),
            ]
        if "TaskRecognition" in message_type:
            return [
                float(message.task),
                float(message.unit_task),
                float(message.confidence),
            ]
        if message_type == "std_msgs/msg/Bool":
            return [float(bool(message.data))]
        if "CameraInfo" in message_type:
            distortion = [float(item) for item in list(message.d)[:8]]
            while len(distortion) < 8:
                distortion.append(np.nan)
            values = [
                float(message.width),
                float(message.height),
            ]
            values.extend(distortion)
            values.extend(float(item) for item in message.k)
            values.extend(float(item) for item in message.r)
            values.extend(float(item) for item in message.p)
            return values
        return []

    def _buffer_to_stream(
        self,
        name: str,
        message_type: str,
        buffer: Dict[str, List],
        original_message_count: int,
        sample_stride: int,
    ) -> Optional[DataStream]:
        timestamps = np.asarray(buffer["t"], dtype=float)
        if len(timestamps) == 0:
            return None
        if _is_ros_image_message_type(message_type):
            return DataStream(
                name,
                "image",
                timestamps,
                image_bytes=buffer["img"],
                source_type=message_type,
                original_message_count=original_message_count,
                sample_stride=sample_stride,
            )
        values = self._pad_rows(buffer["v"])
        labels = self._labels_for_type(message_type, values.shape[1])
        return DataStream(
            name,
            "timeseries",
            timestamps,
            values,
            labels=labels,
            source_type=message_type,
            original_message_count=original_message_count,
            sample_stride=sample_stride,
        )

    def _pad_rows(self, rows: List[List[float]]) -> np.ndarray:
        max_len = max(len(row) for row in rows)
        array = np.full((len(rows), max_len), np.nan, dtype=float)
        for row_idx, row in enumerate(rows):
            array[row_idx, :len(row)] = row
        return array

    def _labels_for_type(self, message_type: str, width: int) -> List[str]:
        if "WrenchStamped" in message_type:
            return ["fx", "fy", "fz", "tx", "ty", "tz"]
        if "PoseStamped" in message_type:
            return ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if "JointState" in message_type:
            base = [f"pos_{idx}" for idx in range(7)]
            base += [f"vel_{idx}" for idx in range(7)]
            base += [f"eff_{idx}" for idx in range(7)]
            return base[:width]
        if "GripperStates" in message_type:
            return ["status", "object_status", "position", "current"][:width]
        if "MasterStates" in message_type:
            base = [
                "raw_position_x",
                "raw_position_y",
                "raw_position_z",
                "raw_rpy_roll",
                "raw_rpy_pitch",
                "raw_rpy_yaw",
                "delta_position_x",
                "delta_position_y",
                "delta_position_z",
                "delta_orientation_x",
                "delta_orientation_y",
                "delta_orientation_z",
                "delta_orientation_w",
                "is_indexing",
            ]
            return base[:width]
        if "TaskRecognition" in message_type:
            return ["task", "unit_task", "confidence"][:width]
        if message_type == "std_msgs/msg/Bool":
            return ["data"][:width]
        if "CameraInfo" in message_type:
            base = ["width", "height"]
            base += [f"d_{idx}" for idx in range(8)]
            base += [f"k_{idx}" for idx in range(9)]
            base += [f"r_{idx}" for idx in range(9)]
            base += [f"p_{idx}" for idx in range(12)]
            return base[:width]
        custom_labels = self.custom_parser.labels_for_type(message_type)
        if custom_labels:
            return custom_labels[:width]
        return [f"value_{idx}" for idx in range(width)]

    def _finalize_streams(self, session: DatasetSession) -> None:
        stream_starts = [
            float(stream.timestamps[0])
            for stream in session.streams.values()
            if len(stream.timestamps)
        ]
        if stream_starts and session.start_time_sec <= 0.0:
            session.start_time_sec = min(stream_starts)
        for stream in session.streams.values():
            if len(stream.timestamps):
                session.start_time_sec = min(session.start_time_sec, float(stream.timestamps[0]))
                session.end_time_sec = max(session.end_time_sec, float(stream.timestamps[-1]))
        session.duration_sec = max(session.end_time_sec - session.start_time_sec, 1e-6)
        self.progress_callback(100, f"Import completed: {session.file_path.name}")


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode compressed image bytes into RGB numpy array."""
    buffer = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            return np.asarray(pil_image.convert("RGB"))
        except Exception:
            return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def decode_depth_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode PNG-backed depth bytes while preserving native sensor values."""
    png_signature = b"\x89PNG\r\n\x1a\n"
    png_offset = image_bytes.find(png_signature)
    payload = image_bytes[png_offset:] if png_offset >= 0 else image_bytes
    buffer = np.frombuffer(payload, np.uint8)
    depth_image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        return None
    if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
        return depth_image[:, :, 0]
    return depth_image
