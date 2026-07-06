# multrecog_core.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
import h5py
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any

class HDF5DataLoader:
    """
    HDF5 파일 입출력 및 데이터 처리를 담당하는 모델 클래스.
    """
    def __init__(self, file_path: str, label_dataset_name: str, logger_func=print):
        self.file_path = file_path
        self.label_dataset_name = label_dataset_name
        self.h5file = h5py.File(file_path, 'r')
        self.stream_names: List[str] = []
        self.frame_count = 0
        self.logger = logger_func

        def _find_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.stream_names.append(name)

        self.h5file.visititems(_find_datasets)
        
        for name in self.stream_names:
            if name != self.label_dataset_name:
                try:
                    dset = self.h5file[name]
                    if dset.shape:
                        self.frame_count = max(self.frame_count, dset.shape[0])
                except Exception:
                    pass

    def get_stream_names(self) -> List[str]:
        return [n for n in self.stream_names if n != self.label_dataset_name]

    def get_frame_count(self) -> int:
        return self.frame_count

    def get_data(self, stream_name: str) -> Optional[np.ndarray]:
        if stream_name in self.h5file:
            return self.h5file[stream_name][:]
        return None

    def check_frame0_validity(self, image_data: Optional[np.ndarray]) -> bool:
        """0번 프레임 이미지 데이터가 유효한지 검사합니다."""
        if image_data is None or len(image_data) == 0:
            return False
        
        # 압축 이미지(object/bytes)인 경우 디코딩 테스트
        if image_data.dtype == np.object_ or image_data.dtype.type in (np.string_, np.bytes_):
            try:
                buf = np.frombuffer(image_data[0], np.uint8)
                if buf.size == 0: return True # Bad
                if cv2.imdecode(buf, cv2.IMREAD_COLOR) is None: return True # Bad
            except Exception:
                return True # Bad
        return False # Good

    def load_labels(self) -> List[Tuple[int, int, List[int]]]:
        if self.label_dataset_name not in self.h5file:
            return []

        dset = self.h5file[self.label_dataset_name]
        
        # [수정] 1차원 배열(N,)과 2차원 배열(N, 2) 모두 지원하도록 조건 완화
        is_1d = (dset.ndim == 1)
        is_2d = (dset.ndim == 2 and dset.shape[1] == 2)

        if not (is_1d or is_2d):
            self.logger(f"Unsupported label format: {dset.shape}. Expected (N,) or (N, 2).", "WARN")
            return []

        segments = []
        if len(dset) == 0: return segments

        start_frame = 0
        
        # [수정] 데이터 차원에 따라 초기 라벨 값 설정
        if is_1d:
            # 1차원이면 두 번째 클래스는 없는 것(-1)으로 처리
            current_labels = (dset[0], -1)
        else:
            current_labels = tuple(dset[0])

        for i in range(1, len(dset)):
            if is_1d:
                new_labels = (dset[i], -1)
            else:
                new_labels = tuple(dset[i])
                
            if new_labels != current_labels:
                # numpy 타입을 int로 변환하여 저장
                c_ids = [int(l) for l in current_labels if l != -1]
                if c_ids: segments.append((start_frame, i - 1, sorted(c_ids)))
                start_frame = i
                current_labels = new_labels
        
        # 마지막 세그먼트 처리
        c_ids = [int(l) for l in current_labels if l != -1]
        if c_ids: segments.append((start_frame, len(dset) - 1, sorted(c_ids)))
        
        return segments

    def export_with_labels(self, output_path: str, segments: List[Tuple[int, int, List[int]]]):
        with h5py.File(output_path, 'w') as f_out:
            for name in self.get_stream_names():
                self.h5file.copy(name, f_out, name=name)
            
            total = self.get_frame_count()
            if total > 0:
                # 내보낼 때는 항상 표준 포맷인 (N, 2)로 저장
                labels = np.full((total, 2), -1, dtype=np.int32)
                for start, end, c_ids in segments:
                    sorted_ids = sorted(c_ids)
                    if len(sorted_ids) >= 1: labels[start:end+1, 0] = sorted_ids[0]
                    if len(sorted_ids) >= 2: labels[start:end+1, 1] = sorted_ids[1]
                f_out.create_dataset(self.label_dataset_name, data=labels)
            else:
                f_out.create_dataset(self.label_dataset_name, shape=(0, 2), dtype=np.int32)

    def close(self):
        self.h5file.close()


class LabelingLogic:
    """
    라벨링 상태 및 세그먼트 연산 로직 클래스.
    """
    def __init__(self, color_palette: List[Any]):
        self.segments: List[Tuple[int, int, List[int]]] = []
        self.is_labeling = False
        self.label_start_frame: Optional[int] = None
        self.class_colors: Dict[int, Any] = {}
        self.color_palette = color_palette
        self.next_color_index = 0

    def reset(self):
        self.segments = []
        self.is_labeling = False
        self.label_start_frame = None
        self.class_colors = {}
        self.next_color_index = 0

    def start_labeling(self, current_frame: int):
        self.is_labeling = True
        self.label_start_frame = current_frame

    def stop_labeling(self, end_frame: int, class_ids: List[int]):
        self.is_labeling = False
        if self.label_start_frame is None or not class_ids: return
        
        start = min(self.label_start_frame, end_frame)
        end = max(self.label_start_frame, end_frame)
        self._add_segment((start, end, sorted(list(set(class_ids)))[:2]))

    def _add_segment(self, new_segment: Tuple[int, int, List[int]]):
        new_s, new_e, new_c = new_segment
        updated = []
        for old_s, old_e, old_c in self.segments:
            if old_e < new_s or old_s > new_e:
                updated.append((old_s, old_e, old_c))
                continue
            if old_s < new_s: updated.append((old_s, new_s - 1, old_c))
            if old_e > new_e: updated.append((new_e + 1, old_e, old_c))
        updated.append(new_segment)
        self.segments = sorted(updated)

    def edit_segment(self, index: int, new_data: Tuple[int, int, List[int]]):
        self.segments.pop(index)
        self._add_segment(new_data)

    def delete_segment(self, index: int):
        return self.segments.pop(index)

    def get_class_color(self, class_id: int, temp: bool = False):
        if temp: return self.color_palette[len(self.class_colors) % len(self.color_palette)]
        if class_id not in self.class_colors:
            self.class_colors[class_id] = self.color_palette[self.next_color_index % len(self.color_palette)]
            self.next_color_index += 1
        return self.class_colors[class_id]

    def get_class_at(self, frame_index: int) -> str:
        for s, e, c in self.segments:
            if s <= frame_index <= e: return ", ".join(map(str, c))
        return "None"
