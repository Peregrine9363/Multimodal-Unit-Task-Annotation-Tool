# media_label_exporter.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Class-organized MP4 and image-sequence export
# ==============================================================================

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import yaml

from data_models import DatasetSession
from labeling_io import LabelSegment
from media_sources import IMAGE_EXTENSIONS


METADATA_FIELDS = (
    "relative_path",
    "media_type",
    "class_id",
    "class_name",
    "source_path",
    "start_frame",
    "end_frame",
    "frame_index",
)
ProgressCallback = Callable[[int, str], None]


@dataclass
class MediaLabelExportConfig:
    """Validated media label export settings."""

    mode: str = "basic"
    output_dir_name: str = "label"
    labels_file_suffix: str = "_labels.csv"
    metadata_file_name: str = "metadata.csv"
    class_folder_template: str = "class_{class_id}"
    class_names: Optional[Dict] = None
    duplicate_multilabel_segments: bool = True
    video_codec: str = "mp4v"
    image_segment_template: str = "{source}_{start:06d}_{end:06d}"
    hdf5_mode: str = "embedded"
    hdf5_output_name_suffix: str = "_labeled"
    hdf5_export_csv_sidecar: bool = False

    def __post_init__(self) -> None:
        self.class_names = dict(self.class_names or {})

    @property
    def split_enabled(self) -> bool:
        return self.mode == "split"


@dataclass
class MediaExportResult:
    """Summary of one split-media export."""

    exported_files: int
    metadata_path: Path


def load_media_label_export_config(path: Path) -> MediaLabelExportConfig:
    """Load basic/split export behavior from YAML."""
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError("Media label export config root must be a mapping.")
    mode = str(data.get("mode", "basic")).strip().lower()
    if mode not in ("basic", "split"):
        raise ValueError("Media label export mode must be 'basic' or 'split'.")
    split = data.get("split", {}) or {}
    if not isinstance(split, dict):
        raise ValueError("Media label export split config must be a mapping.")
    hdf5_config = data.get("hdf5", {}) or {}
    if not isinstance(hdf5_config, dict):
        raise ValueError("Media label export hdf5 config must be a mapping.")
    hdf5_mode = str(hdf5_config.get("mode", "embedded")).strip().lower()
    if hdf5_mode not in ("embedded", "csv"):
        raise ValueError("Media label export hdf5.mode must be 'embedded' or 'csv'.")
    hdf5_name_suffix = str(
        hdf5_config.get("output_name_suffix", "_labeled")
    ).strip()
    has_path_separator = "/" in hdf5_name_suffix or "\\" in hdf5_name_suffix
    if (
        not hdf5_name_suffix
        or has_path_separator
        or Path(hdf5_name_suffix).name != hdf5_name_suffix
    ):
        raise ValueError("hdf5.output_name_suffix must be a plain file-name suffix.")
    codec = str(split.get("video_codec", "mp4v"))
    if len(codec) != 4:
        raise ValueError("split.video_codec must contain exactly four characters.")
    return MediaLabelExportConfig(
        mode=mode,
        output_dir_name=str(data.get("output_dir_name", "label")),
        labels_file_suffix=str(data.get("labels_file_suffix", "_labels.csv")),
        metadata_file_name=str(split.get("metadata_file_name", "metadata.csv")),
        class_folder_template=str(
            split.get("class_folder_template", "class_{class_id}")
        ),
        class_names=split.get("class_names", {}) or {},
        duplicate_multilabel_segments=bool(
            split.get("duplicate_multilabel_segments", True)
        ),
        video_codec=codec,
        image_segment_template=str(
            split.get(
                "image_segment_template",
                "{source}_{start:06d}_{end:06d}",
            )
        ),
        hdf5_mode=hdf5_mode,
        hdf5_output_name_suffix=hdf5_name_suffix,
        hdf5_export_csv_sidecar=bool(
            hdf5_config.get("export_csv_sidecar", False)
        ),
    )


class MediaSegmentExporter:
    """Export labeled MP4 or image intervals into class folders."""

    def __init__(
        self,
        config: MediaLabelExportConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.progress_callback = progress_callback

    def export(
        self,
        session: DatasetSession,
        segments: List[LabelSegment],
        export_root: Path,
    ) -> MediaExportResult:
        self._report(2, "Preparing split media export...")
        export_root.mkdir(parents=True, exist_ok=True)
        source_path = session.file_path.expanduser().resolve()
        metadata_rows = []
        if session.source_kind == "mp4":
            metadata_rows = self._export_video_segments(
                source_path,
                segments,
                export_root,
            )
        elif session.source_kind in ("image", "image_sequence"):
            metadata_rows = self._export_image_segments(
                session,
                segments,
                export_root,
            )
        else:
            raise ValueError(
                "Split mode supports MP4, a single image, or an image folder."
            )
        self._report(95, "Writing media metadata...")
        metadata_path = export_root / self.config.metadata_file_name
        self._write_metadata(metadata_path, source_path, metadata_rows)
        self._report(100, "Split media export completed.")
        return MediaExportResult(len(metadata_rows), metadata_path)

    def _export_video_segments(
        self,
        source_path: Path,
        segments: List[LabelSegment],
        export_root: Path,
    ) -> List[Dict[str, object]]:
        rows = []
        tasks = self._video_export_tasks(segments)
        for task_index, (start, end, class_id) in enumerate(tasks):
            class_name = self._class_name(class_id)
            class_dir = export_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{self._safe_name(source_path.stem)}_"
                f"{start:06d}_{end:06d}.mp4"
            )
            output_path = class_dir / filename
            self._write_video_segment(
                source_path,
                output_path,
                start,
                end,
                task_index,
                len(tasks),
            )
            rows.append(self._metadata_row(
                output_path,
                export_root,
                "video",
                class_id,
                class_name,
                source_path,
                start,
                end,
            ))
        if not tasks:
            self._report(90, "No labeled video segments to export.")
        return rows

    def _video_export_tasks(
        self,
        segments: List[LabelSegment],
    ) -> List[Tuple[int, int, int]]:
        tasks = []
        for start, end, class_ids in segments:
            tasks.extend(
                (start, end, class_id)
                for class_id in self._target_class_ids(class_ids)
            )
        return tasks

    def _write_video_segment(
        self,
        source_path: Path,
        output_path: Path,
        start: int,
        end: int,
        task_index: int,
        task_count: int,
    ) -> None:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open source video: {source_path}")
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        start, end = self._clamp_range(start, end, frame_count)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0.0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Failed to create split video: {output_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        written_frames = 0
        try:
            for frame_index in range(start, end + 1):
                success, frame = capture.read()
                if not success or frame is None:
                    break
                writer.write(frame)
                written_frames += 1
                self._report_video_progress(
                    output_path.name,
                    frame_index - start + 1,
                    end - start + 1,
                    task_index,
                    task_count,
                )
        finally:
            writer.release()
            capture.release()
        expected_frames = end - start + 1
        if written_frames != expected_frames:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"Split video is incomplete: expected={expected_frames}, "
                f"written={written_frames}, path={output_path}"
            )

    def _export_image_segments(
        self,
        session: DatasetSession,
        segments: List[LabelSegment],
        export_root: Path,
    ) -> List[Dict[str, object]]:
        image_paths = self._image_source_paths(session)
        rows = []
        source_path = session.file_path.expanduser().resolve()
        total_copies = self._image_copy_count(segments, len(image_paths))
        copied_count = 0
        for start, end, class_ids in segments:
            start, end = self._clamp_range(start, end, len(image_paths))
            for class_id in self._target_class_ids(class_ids):
                class_name = self._class_name(class_id)
                segment_name = self.config.image_segment_template.format(
                    source=self._safe_name(source_path.stem),
                    start=start,
                    end=end,
                    class_id=class_id,
                )
                segment_dir = export_root / class_name / self._safe_name(segment_name)
                segment_dir.mkdir(parents=True, exist_ok=True)
                for frame_index in range(start, end + 1):
                    input_path = image_paths[frame_index]
                    output_path = segment_dir / (
                        f"{frame_index:06d}_{input_path.name}"
                    )
                    shutil.copy2(input_path, output_path)
                    copied_count += 1
                    self._report_copy_progress(
                        output_path.name,
                        copied_count,
                        total_copies,
                    )
                    rows.append(self._metadata_row(
                        output_path,
                        export_root,
                        "image",
                        class_id,
                        class_name,
                        source_path,
                        start,
                        end,
                        frame_index,
                    ))
        if total_copies == 0:
            self._report(90, "No labeled images to export.")
        return rows

    def _image_copy_count(
        self,
        segments: List[LabelSegment],
        frame_count: int,
    ) -> int:
        total = 0
        for start, end, class_ids in segments:
            start, end = self._clamp_range(start, end, frame_count)
            total += (end - start + 1) * len(self._target_class_ids(class_ids))
        return total

    def _image_source_paths(self, session: DatasetSession) -> List[Path]:
        for stream in session.streams.values():
            if stream.stream_type != "image":
                continue
            paths = getattr(stream.image_bytes, "paths", None)
            if paths:
                return [Path(path).resolve() for path in paths]
        source_path = session.file_path.expanduser().resolve()
        if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [source_path]
        raise ValueError("No source image paths are available for split export.")

    def _target_class_ids(self, class_ids: List[int]) -> List[int]:
        unique_ids = sorted(set(int(class_id) for class_id in class_ids))
        if self.config.duplicate_multilabel_segments:
            return unique_ids
        return unique_ids[:1]

    def _class_name(self, class_id: int) -> str:
        configured = self.config.class_names.get(
            str(class_id),
            self.config.class_names.get(class_id),
        )
        if configured is None:
            configured = self.config.class_folder_template.format(
                class_id=class_id
            )
        return self._safe_name(str(configured))

    def _metadata_row(
        self,
        output_path: Path,
        export_root: Path,
        media_type: str,
        class_id: int,
        class_name: str,
        source_path: Path,
        start: int,
        end: int,
        frame_index: Optional[int] = None,
    ) -> Dict[str, object]:
        return {
            "relative_path": output_path.relative_to(export_root).as_posix(),
            "media_type": media_type,
            "class_id": class_id,
            "class_name": class_name,
            "source_path": str(source_path),
            "start_frame": start,
            "end_frame": end,
            "frame_index": "" if frame_index is None else frame_index,
        }

    def _write_metadata(
        self,
        metadata_path: Path,
        source_path: Path,
        new_rows: List[Dict[str, object]],
    ) -> None:
        rows = []
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        source_text = str(source_path)
        rows = [row for row in rows if row.get("source_path") != source_text]
        rows.extend(new_rows)
        rows.sort(key=lambda row: (
            str(row.get("class_id", "")),
            str(row.get("relative_path", "")),
        ))
        with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _report_video_progress(
        self,
        filename: str,
        written_frames: int,
        total_frames: int,
        task_index: int,
        task_count: int,
    ) -> None:
        if total_frames <= 0 or task_count <= 0:
            return
        interval = max(total_frames // 100, 1)
        if written_frames % interval and written_frames < total_frames:
            return
        task_ratio = min(float(written_frames) / float(total_frames), 1.0)
        overall_ratio = (float(task_index) + task_ratio) / float(task_count)
        self._report(
            5 + int(85.0 * overall_ratio),
            f"Writing video segment: {filename}",
        )

    def _report_copy_progress(
        self,
        filename: str,
        copied_count: int,
        total_copies: int,
    ) -> None:
        if total_copies <= 0:
            return
        interval = max(total_copies // 100, 1)
        if copied_count % interval and copied_count < total_copies:
            return
        ratio = min(float(copied_count) / float(total_copies), 1.0)
        self._report(
            5 + int(85.0 * ratio),
            f"Copying labeled image: {filename}",
        )

    def _report(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(percent, message)

    @staticmethod
    def _clamp_range(start: int, end: int, frame_count: int) -> Tuple[int, int]:
        if frame_count <= 0:
            raise ValueError("The source media contains no frames.")
        clamped_start = max(0, min(int(start), frame_count - 1))
        clamped_end = max(0, min(int(end), frame_count - 1))
        if clamped_start > clamped_end:
            raise ValueError(f"Invalid media segment: {start}..{end}")
        return clamped_start, clamped_end

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^\w.-]+", "_", value.strip())
        return safe.strip("._") or "unnamed"
