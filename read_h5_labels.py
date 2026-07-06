# read_h5_labels.py
import h5py
import os
import numpy as np

# =================================================================================================
# 핵심 파라미터 (Core Parameters)
# =================================================================================================
# 라벨 데이터를 읽어올 HDF5 파일의 경로를 지정합니다.
H5_FILE_PATH = "dataset/l-line_cable_3/label/001_labeled.hdf5"

# 읽어올 데이터셋의 이름을 지정합니다.
LABEL_DATASET_NAME = "labels"

# =================================================================================================
# 라벨 데이터 읽기 함수 (Label Reading Function)
# =================================================================================================
def read_labels_from_h5(file_path: str, dataset_name: str):
    """
    HDF5 파일에서 프레임별 클래스 ID가 저장된 1차원 데이터셋을 읽어
    그 내용을 목록 형태로 출력합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 경로에 파일이 존재하지 않습니다.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            # 2. 파일 내에 해당 데이터셋이 있는지 확인
            if dataset_name not in f:
                print(f"오류: '{file_path}' 파일에 '{dataset_name}' 데이터셋이 존재하지 않습니다.")
                return

            print(f"===== HDF5 파일의 '{dataset_name}' 데이터 내용 (프레임별) =====")
            print(f"File: '{os.path.basename(file_path)}'")
            print("-" * 50)

            # 3. 데이터셋을 NumPy 배열로 읽어오기
            labels_data = f[dataset_name][:]

            # 4. 데이터셋이 1차원 배열이 맞는지 확인
            if labels_data.ndim != 1:
                print(f"오류: 이 스크립트는 1차원 배열 형태의 라벨 데이터만 지원합니다.")
                print(f"실제 데이터 차원: {labels_data.ndim}")
                return

            # 5. 데이터가 비어있는지 확인
            if labels_data.shape[0] == 0:
                print("데이터셋이 비어 있습니다.")
            else:
                # ================== [ 수정된 부분 ] ==================
                # enumerate를 사용하여 각 프레임의 인덱스와 클래스 ID를 함께 가져옵니다.
                for frame_index, class_id in enumerate(labels_data):
                    print(f"- 프레임 {frame_index:04d}: 클래스 ID = {class_id}")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            print("-" * 50)
            print(f"총 {len(labels_data)}개의 프레임 데이터를 읽었습니다.")
            print("==================== 출력 완료 ====================")

    except Exception as e:
        print(f"파일을 열거나 데이터를 읽는 중 오류가 발생했습니다: {e}")


# =================================================================================================
# 메인 실행 블록 (Main Execution Block)
# =================================================================================================
if __name__ == '__main__':
    read_labels_from_h5(H5_FILE_PATH, LABEL_DATASET_NAME)