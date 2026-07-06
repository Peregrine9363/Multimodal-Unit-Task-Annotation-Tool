# custom_msg_parser.py
# ==============================================================================
# CDR fallback parser for local msg definitions
# ==============================================================================

import struct
from typing import List, Optional, Tuple


class CdrReader:
    """Minimal little-endian CDR reader for known local message previews."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 4  # Skip CDR encapsulation header.

    def read_bool(self) -> float:
        value = self.read_uint8()
        return float(value != 0)

    def read_uint8(self) -> int:
        value = self.data[self.offset]
        self.offset += 1
        return value

    def read_int32(self) -> int:
        self._align(4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_uint32(self) -> int:
        self._align(4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_float64(self) -> float:
        self._align(8)
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return value

    def read_float32(self) -> float:
        self._align(4)
        value = struct.unpack_from("<f", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_string(self) -> str:
        length = self.read_uint32()
        raw = self.data[self.offset:self.offset + length]
        self.offset += length
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def read_header(self) -> Tuple[int, int, str]:
        sec = self.read_int32()
        nanosec = self.read_uint32()
        frame_id = self.read_string()
        return sec, nanosec, frame_id

    def read_point3(self) -> List[float]:
        return [self.read_float64(), self.read_float64(), self.read_float64()]

    def read_pose7(self) -> List[float]:
        position = self.read_point3()
        orientation = [
            self.read_float64(),
            self.read_float64(),
            self.read_float64(),
            self.read_float64(),
        ]
        return position + orientation

    def _align(self, size: int) -> None:
        remainder = (self.offset - 4) % size
        if remainder:
            self.offset += size - remainder


class LocalCustomMessageParser:
    """Decode local custom messages described by the workspace doc/*.msg files."""

    def decode_values(self, message_type: str, raw_data: bytes) -> Optional[List[float]]:
        if message_type == "shared_msg/msg/GripperStates":
            return self._decode_gripper_states(raw_data)
        if message_type == "shared_msg/msg/MasterStates":
            return self._decode_master_states(raw_data)
        if message_type == "shared_msg/msg/TaskRecognition":
            return self._decode_task_recognition(raw_data)
        return None

    def labels_for_type(self, message_type: str) -> List[str]:
        if message_type == "shared_msg/msg/GripperStates":
            return ["status", "object_status", "position", "current"]
        if message_type == "shared_msg/msg/MasterStates":
            return [
                "raw_x",
                "raw_y",
                "raw_z",
                "raw_roll",
                "raw_pitch",
                "raw_yaw",
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_qx",
                "delta_qy",
                "delta_qz",
                "delta_qw",
                "is_indexing",
            ]
        if message_type == "shared_msg/msg/TaskRecognition":
            return ["task", "unit_task", "confidence"]
        return []

    def _decode_gripper_states(self, raw_data: bytes) -> Optional[List[float]]:
        try:
            reader = CdrReader(raw_data)
            reader.read_header()
            return [
                float(reader.read_uint8()),
                float(reader.read_uint8()),
                float(reader.read_uint8()),
                float(reader.read_uint8()),
            ]
        except Exception:
            return None

    def _decode_master_states(self, raw_data: bytes) -> Optional[List[float]]:
        try:
            reader = CdrReader(raw_data)
            reader.read_header()
            raw_position = reader.read_point3()
            raw_rpy = reader.read_point3()
            delta_pose = reader.read_pose7()
            is_indexing = [reader.read_bool()]
            return raw_position + raw_rpy + delta_pose + is_indexing
        except Exception:
            return None

    def _decode_task_recognition(self, raw_data: bytes) -> Optional[List[float]]:
        try:
            reader = CdrReader(raw_data)
            reader.read_header()
            return [
                float(reader.read_uint8()),
                float(reader.read_uint8()),
                float(reader.read_float32()),
            ]
        except Exception:
            return None
