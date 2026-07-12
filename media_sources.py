# media_sources.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Lazy image payload sequences for folders and video files
# ==============================================================================

import re
from collections.abc import Sequence
from pathlib import Path
from typing import List, Union

import cv2


IMAGE_EXTENSIONS = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)


def natural_path_key(path: Path) -> tuple:
    """Sort paths naturally so frame_2 precedes frame_10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_image_sequence(folder: Path) -> List[Path]:
    """Return directly contained image files in deterministic frame order."""
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        return []
    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_path_key)


class ImageFolderSequence(Sequence[bytes]):
    """Read image payloads lazily from one ordered folder sequence."""

    def __init__(self, paths: List[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        return self.paths[self._normalized_index(index)].read_bytes()

    def _normalized_index(self, index: int) -> int:
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("Image frame index out of range")
        return normalized


class VideoFrameSequence(Sequence[bytes]):
    """Decode requested video frames lazily and expose PNG image payloads."""

    def __init__(self, video_path: Path):
        self.video_path = video_path.expanduser().resolve()
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        self.frame_count = self._read_frame_count()
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cached_index = -1
        self._cached_payload = b""

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = self._normalized_index(index)
        if index == self._cached_index:
            return self._cached_payload
        frame = self._decode_frame(index)
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise ValueError(
                f"Failed to encode video preview frame {index}: {self.video_path}"
            )
        self._cached_index = index
        self._cached_payload = encoded.tobytes()
        return self._cached_payload

    def close(self) -> None:
        """Release the OpenCV video handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_frame_count(self) -> int:
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return frame_count
        count = 0
        while self._capture.grab():
            count += 1
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if count <= 0:
            raise ValueError(f"Video contains no readable frames: {self.video_path}")
        return count

    def _normalized_index(self, index: int) -> int:
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("Video frame index out of range")
        return normalized

    def _decode_frame(self, index: int):
        current_index = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current_index != index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self._capture.read()
        if not success or frame is None:
            raise ValueError(
                f"Failed to decode video frame {index}: {self.video_path}"
            )
        return frame

    def __del__(self):
        self.close()
