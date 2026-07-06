# data_models.py
# ==============================================================================
# Shared data structures
# ==============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TopicInfo:
    """Metadata for a rosbag topic or imported dataset stream."""

    name: str
    message_type: str
    message_count: int
    data_size_bytes: int = 0
    serialization_format: str = ""


@dataclass
class DataStream:
    """Preview stream used by the GUI data views."""

    name: str
    stream_type: str
    timestamps: np.ndarray
    values: Optional[np.ndarray] = None
    image_bytes: List[bytes] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    source_type: str = ""
    original_message_count: int = 0
    sample_stride: int = 1

    def is_empty(self) -> bool:
        return len(self.timestamps) == 0

    def nearest_index(self, timestamp_sec: float) -> int:
        if self.is_empty():
            return 0
        idx = int(np.searchsorted(self.timestamps, timestamp_sec))
        if idx <= 0:
            return 0
        if idx >= len(self.timestamps):
            return len(self.timestamps) - 1
        prev_delta = abs(timestamp_sec - self.timestamps[idx - 1])
        next_delta = abs(self.timestamps[idx] - timestamp_sec)
        return idx - 1 if prev_delta <= next_delta else idx


@dataclass
class DatasetSession:
    """Current imported dataset state."""

    file_path: Path
    workspace_path: Path
    file_list: List[Path]
    file_index: int
    topic_info: List[TopicInfo] = field(default_factory=list)
    streams: Dict[str, DataStream] = field(default_factory=dict)
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0
    message_count: int = 0
    note_text: str = ""
    source_kind: str = ""

    @property
    def stream_names(self) -> List[str]:
        return sorted(self.streams.keys())

    def has_stream(self, name: str) -> bool:
        return name in self.streams

    def get_stream(self, name: str) -> Optional[DataStream]:
        return self.streams.get(name)

    def timestamp_from_ratio(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return self.start_time_sec + self.duration_sec * ratio

    def ratio_from_timestamp(self, timestamp_sec: float) -> float:
        if self.duration_sec <= 0.0:
            return 0.0
        return (timestamp_sec - self.start_time_sec) / self.duration_sec
