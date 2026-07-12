# labeling_io.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Label segment import/export helpers
# ==============================================================================

import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np


LabelSegment = Tuple[int, int, List[int]]
ProgressCallback = Callable[[int, str], None]


@dataclass
class LabelImportResult:
    """라벨 파일에서 읽은 segment와 보조 메타데이터입니다."""

    segments: List[LabelSegment]
    total_frames: int
    source_path: str = ""
    source_format: str = ""
    label_domain: str = "frame_index"
    timestamp_bounds: Optional[Tuple[float, float]] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class LabelStorage:
    """공통 CSV export와 CSV/기존 HDF5 import를 제공합니다."""

    def __init__(self, label_dataset_name: str, logger=print):
        self.label_dataset_name = label_dataset_name
        self.logger = logger

    def load_from_hdf5(
        self,
        file_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[LabelSegment]:
        """기존 HDF5 파일 안의 frame-wise label dataset을 segment 목록으로 변환합니다."""
        self._report(progress_callback, 5, "Checking embedded HDF5 labels...")
        with h5py.File(file_path, "r") as h5_file:
            has_frame_labels = self.label_dataset_name in h5_file
            has_segments = "label_segments" in h5_file
            if not has_frame_labels and not has_segments:
                return []
        return self.import_labels(file_path, progress_callback).segments

    def import_labels(
        self,
        file_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> LabelImportResult:
        """CSV 또는 기존 HDF5 라벨 파일을 읽습니다."""
        if file_path.suffix.lower() == ".csv":
            return self._import_csv(file_path, progress_callback)
        self._report(progress_callback, 10, "Opening label HDF5 file...")
        with h5py.File(file_path, "r") as h5_file:
            self._report(progress_callback, 35, "Reading label metadata...")
            metadata = self._read_metadata(h5_file)
            self._report(progress_callback, 60, "Reconstructing label segments...")
            segments, total_frames = self._read_label_payload(h5_file)
            timestamp_bounds = self._read_timestamp_bounds(h5_file, total_frames)
            result = LabelImportResult(
                segments=segments,
                total_frames=total_frames,
                source_path=metadata.get("source_path", ""),
                source_format=metadata.get("source_format", ""),
                label_domain=metadata.get("label_domain", "frame_index"),
                timestamp_bounds=timestamp_bounds,
                metadata=metadata,
            )
        self._report(progress_callback, 100, "Label import completed.")
        return result

    def export(
        self,
        source_path: Path,
        output_path: Path,
        segments: List[LabelSegment],
        total_frames: int,
        timestamp_bounds: Optional[Tuple[float, float]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """모든 입력 포맷의 frame-wise 라벨을 단일 CSV로 저장합니다."""
        self._export_csv(
            output_path,
            segments,
            total_frames,
            timestamp_bounds,
            progress_callback,
        )

    def _import_csv(
        self,
        file_path: Path,
        progress_callback: Optional[ProgressCallback],
    ) -> LabelImportResult:
        """Read a frame-wise label CSV and reconstruct contiguous segments."""
        self._report(progress_callback, 5, "Counting label CSV rows...")
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            total_rows = max(sum(1 for _line in csv_file) - 1, 0)
        self._report(progress_callback, 10, "Reading label CSV...")
        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            self._validate_csv_columns(reader.fieldnames)
            rows = self._read_csv_rows(reader, total_rows, progress_callback)
        if not rows:
            self._report(progress_callback, 100, "Label import completed.")
            return LabelImportResult([], 0, source_format="csv")

        self._report(progress_callback, 55, "Validating label frame indices...")
        frame_indices = [self._csv_int(row, "frame_index") for row in rows]
        if min(frame_indices) < 0:
            raise ValueError("CSV frame_index values must be non-negative.")
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("CSV contains duplicate frame_index values.")

        total_frames = max(frame_indices) + 1
        labels = np.full((total_frames, 2), -1, dtype=np.int32)
        timestamps = []
        for row_index, (frame_index, row) in enumerate(zip(frame_indices, rows)):
            labels[frame_index, 0] = self._csv_int(row, "class_id_1", -1)
            labels[frame_index, 1] = self._csv_int(row, "class_id_2", -1)
            timestamps.append(float(row["timestamp_sec"]))
            self._report_loop_progress(
                progress_callback,
                row_index,
                len(rows),
                60,
                85,
                "Building label timeline...",
            )
        finite_times = [value for value in timestamps if np.isfinite(value)]
        timestamp_bounds = None
        if finite_times:
            timestamp_bounds = (min(finite_times), max(finite_times))
        self._report(progress_callback, 90, "Reconstructing label segments...")
        result = LabelImportResult(
            segments=self._dataset_to_segments(labels),
            total_frames=total_frames,
            source_format="csv",
            label_domain="frame_index",
            timestamp_bounds=timestamp_bounds,
        )
        self._report(progress_callback, 100, "Label import completed.")
        return result

    def _validate_csv_columns(self, field_names: Optional[List[str]]) -> None:
        """Validate the public frame-wise CSV label schema."""
        required = {
            "frame_index",
            "timestamp_sec",
            "class_id_1",
            "class_id_2",
        }
        missing = required.difference(field_names or [])
        if missing:
            raise ValueError(
                "Label CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

    def _read_csv_rows(
        self,
        reader: csv.DictReader,
        total_rows: int,
        progress_callback: Optional[ProgressCallback],
    ) -> List[Dict[str, str]]:
        """Read CSV rows while reporting bounded progress."""
        rows = []
        for row_index, row in enumerate(reader):
            rows.append(row)
            self._report_loop_progress(
                progress_callback,
                row_index,
                total_rows,
                10,
                50,
                "Reading label CSV...",
            )
        return rows

    @staticmethod
    def _report_loop_progress(
        callback: Optional[ProgressCallback],
        index: int,
        total: int,
        start_percent: int,
        end_percent: int,
        message: str,
    ) -> None:
        if callback is None or total <= 0:
            return
        interval = max(total // 100, 1)
        if index % interval and index + 1 < total:
            return
        ratio = min(float(index + 1) / float(total), 1.0)
        percent = start_percent + int((end_percent - start_percent) * ratio)
        callback(percent, message)

    @staticmethod
    def _report(
        callback: Optional[ProgressCallback],
        percent: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(percent, message)

    def _csv_int(self, row: Dict[str, str], key: str, default: int = 0) -> int:
        """Parse an integer CSV field while allowing an empty default value."""
        value = str(row.get(key, "")).strip()
        if not value:
            return default
        return int(value)

    def _export_csv(
        self,
        output_path: Path,
        segments: List[LabelSegment],
        total_frames: int,
        timestamp_bounds: Optional[Tuple[float, float]],
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        """Write one frame-wise CSV containing the complete label timeline."""
        self._report(progress_callback, 5, "Preparing frame-wise labels...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels = self._segments_to_frame_labels(segments, total_frames)
        field_names = (
            "frame_index",
            "timestamp_sec",
            "class_id_1",
            "class_id_2",
        )
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for frame_index, frame_labels in enumerate(labels):
                writer.writerow({
                    "frame_index": frame_index,
                    "timestamp_sec": self._index_to_timestamp(
                        frame_index,
                        timestamp_bounds,
                        total_frames,
                    ),
                    "class_id_1": int(frame_labels[0]),
                    "class_id_2": int(frame_labels[1]),
                })
                self._report_loop_progress(
                    progress_callback,
                    frame_index,
                    len(labels),
                    10,
                    100,
                    "Writing label CSV...",
                )
        self._report(progress_callback, 100, "Label CSV export completed.")

    def _dataset_to_segments(self, dataset) -> List[LabelSegment]:
        is_1d = dataset.ndim == 1
        is_2d = dataset.ndim == 2 and dataset.shape[1] == 2
        if not (is_1d or is_2d):
            self.logger(
                f"Unsupported label format: {dataset.shape}. Expected (N,) or (N, 2).",
                "WARN",
            )
            return []
        if len(dataset) == 0:
            return []

        segments: List[LabelSegment] = []
        start_frame = 0
        current_labels = self._labels_at(dataset, 0, is_1d)
        for frame_idx in range(1, len(dataset)):
            new_labels = self._labels_at(dataset, frame_idx, is_1d)
            if new_labels != current_labels:
                self._append_segment(segments, start_frame, frame_idx - 1, current_labels)
                start_frame = frame_idx
                current_labels = new_labels
        self._append_segment(segments, start_frame, len(dataset) - 1, current_labels)
        return segments

    def _read_label_payload(self, h5_file) -> Tuple[List[LabelSegment], int]:
        if self.label_dataset_name in h5_file:
            dataset = h5_file[self.label_dataset_name]
            return self._dataset_to_segments(dataset), len(dataset)
        if "label_segments" in h5_file:
            rows = np.asarray(h5_file["label_segments"][()], dtype=float)
            segments = self._segment_rows_to_segments(rows)
            total_frames = self._infer_total_frames_from_segments(segments)
            return segments, total_frames
        raise ValueError(
            f"Label dataset not found. Expected '{self.label_dataset_name}' "
            "or 'label_segments'."
        )

    def _segment_rows_to_segments(self, rows: np.ndarray) -> List[LabelSegment]:
        rows = np.atleast_2d(rows)
        segments = []
        for row in rows:
            if len(row) < 4:
                continue
            start, end = int(row[0]), int(row[1])
            class_ids = [int(value) for value in row[2:4] if int(value) != -1]
            if start <= end and class_ids:
                segments.append((start, end, sorted(class_ids)))
        return segments

    def _infer_total_frames_from_segments(self, segments: List[LabelSegment]) -> int:
        if not segments:
            return 1
        return max(end for _, end, _ in segments) + 1

    def _read_timestamp_bounds(self, h5_file, total_frames: int) -> Optional[Tuple[float, float]]:
        if "label_segments" in h5_file:
            rows = np.asarray(h5_file["label_segments"][()], dtype=float)
            rows = np.atleast_2d(rows)
            if rows.size and rows.shape[1] >= 6:
                start_sec = float(np.nanmin(rows[:, 4]))
                end_sec = float(np.nanmax(rows[:, 5]))
                if np.isfinite(start_sec) and np.isfinite(end_sec) and end_sec >= start_sec:
                    return start_sec, end_sec
        if total_frames > 1:
            return 0.0, float(total_frames - 1)
        return 0.0, 0.0

    def _read_metadata(self, h5_file) -> Dict[str, str]:
        metadata = {}
        for key, value in h5_file.attrs.items():
            if isinstance(value, bytes):
                metadata[str(key)] = value.decode("utf-8", errors="replace")
            else:
                metadata[str(key)] = str(value)
        return metadata

    def _labels_at(self, dataset, index: int, is_1d: bool) -> Tuple[int, int]:
        if is_1d:
            return int(dataset[index]), -1
        return int(dataset[index][0]), int(dataset[index][1])

    def _append_segment(
        self,
        segments: List[LabelSegment],
        start: int,
        end: int,
        labels: Tuple[int, int],
    ) -> None:
        class_ids = [int(label) for label in labels if int(label) != -1]
        if class_ids:
            segments.append((start, end, sorted(class_ids)))

    def export_labeled_hdf5(
        self,
        source_path: Path,
        output_path: Path,
        segments: List[LabelSegment],
        total_frames: int,
        timestamp_bounds: Optional[Tuple[float, float]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Copy an HDF5 source and replace its embedded labels atomically."""
        source_path = source_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        if source_path.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("Embedded label export requires an HDF5 source file.")
        if source_path == output_path:
            raise ValueError("HDF5 label export must not overwrite the source file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._report(progress_callback, 2, "Preparing labeled HDF5 export...")
        temporary_path = self._temporary_hdf5_path(output_path)
        try:
            self._write_labeled_hdf5(
                source_path,
                temporary_path,
                segments,
                total_frames,
                timestamp_bounds,
                progress_callback,
            )
            temporary_path.replace(output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        self._report(progress_callback, 100, "Labeled HDF5 export completed.")

    def _write_labeled_hdf5(
        self,
        source_path: Path,
        output_path: Path,
        segments: List[LabelSegment],
        total_frames: int,
        timestamp_bounds: Optional[Tuple[float, float]],
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        with h5py.File(source_path, "r") as h5_in, h5py.File(
            output_path,
            "w",
        ) as h5_out:
            self._copy_hdf5_root_attributes(h5_in, h5_out)
            self._copy_hdf5_payload(h5_in, h5_out, progress_callback)
            self._report(progress_callback, 82, "Writing embedded frame labels...")
            labels = self._segments_to_frame_labels(segments, total_frames)
            segment_rows = self._segments_to_rows(
                segments,
                timestamp_bounds,
                total_frames,
            )
            h5_out.create_dataset(self.label_dataset_name, data=labels)
            h5_out.create_dataset("label_segments", data=segment_rows)
            h5_out.attrs["labels_embedded"] = True
            h5_out.attrs["label_dataset"] = self.label_dataset_name
            h5_out.attrs["label_domain"] = "timeline_index"

    def _copy_hdf5_payload(
        self,
        h5_in,
        h5_out,
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        excluded_names = {self.label_dataset_name, "label_segments"}
        names = [name for name in h5_in.keys() if name not in excluded_names]
        if not names:
            self._report(progress_callback, 78, "No source datasets to copy.")
            return
        for index, name in enumerate(names):
            self._report(
                progress_callback,
                5 + int(73.0 * float(index) / float(len(names))),
                f"Copying HDF5 object: {name}",
            )
            h5_in.copy(name, h5_out, name=name)
        self._report(progress_callback, 78, "Source HDF5 data copied.")

    @staticmethod
    def _copy_hdf5_root_attributes(h5_in, h5_out) -> None:
        for key, value in h5_in.attrs.items():
            h5_out.attrs[key] = value

    @staticmethod
    def _temporary_hdf5_path(output_path: Path) -> Path:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}_",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            return Path(temporary_file.name)

    def _export_label_sidecar(
        self,
        source_path: Path,
        output_path: Path,
        segments: List[LabelSegment],
        total_frames: int,
        timestamp_bounds: Optional[Tuple[float, float]],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels = self._segments_to_frame_labels(segments, total_frames)
        segment_rows = self._segments_to_rows(segments, timestamp_bounds, total_frames)
        with h5py.File(output_path, "w") as h5_out:
            h5_out.attrs["source_path"] = str(source_path)
            h5_out.attrs["source_format"] = source_path.suffix.lower().lstrip(".")
            h5_out.attrs["label_domain"] = "timeline_index"
            h5_out.create_dataset(self.label_dataset_name, data=labels)
            h5_out.create_dataset("label_segments", data=segment_rows)

    def _segments_to_frame_labels(
        self,
        segments: List[LabelSegment],
        total_frames: int,
    ) -> np.ndarray:
        labels = np.full((max(total_frames, 0), 2), -1, dtype=np.int32)
        for start, end, class_ids in segments:
            start = max(0, min(start, total_frames - 1))
            end = max(0, min(end, total_frames - 1))
            if start > end:
                continue
            sorted_ids = sorted(class_ids)[:2]
            if sorted_ids:
                labels[start:end + 1, 0] = sorted_ids[0]
            if len(sorted_ids) > 1:
                labels[start:end + 1, 1] = sorted_ids[1]
        return labels

    def _segments_to_rows(
        self,
        segments: List[LabelSegment],
        timestamp_bounds: Optional[Tuple[float, float]],
        total_frames: int,
    ) -> np.ndarray:
        rows = []
        for start, end, class_ids in segments:
            start_sec = self._index_to_timestamp(start, timestamp_bounds, total_frames)
            end_sec = self._index_to_timestamp(end, timestamp_bounds, total_frames)
            ids = sorted(class_ids)[:2]
            row = [start, end, ids[0] if ids else -1, ids[1] if len(ids) > 1 else -1, start_sec, end_sec]
            rows.append(row)
        return np.asarray(rows, dtype=np.float64).reshape((-1, 6))

    def _index_to_timestamp(
        self,
        index: int,
        timestamp_bounds: Optional[Tuple[float, float]],
        total_frames: int,
    ) -> float:
        if not timestamp_bounds or total_frames <= 1:
            return float(index)
        start_sec, end_sec = timestamp_bounds
        ratio = float(index) / float(total_frames - 1)
        return start_sec + (end_sec - start_sec) * ratio
