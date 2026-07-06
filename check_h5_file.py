# check_h5_structure.py
# Copyright 2026 Peregrine9363
# SPDX-License-Identifier: Apache-2.0
import h5py
import os

# =================================================================================================
# 핵심 파라미터 (Core Parameters)
# =================================================================================================
# 분석할 HDF5 파일의 경로를 여기에 직접 입력하세요.
# 예: "dataset/label/002_labeled.hdf5"
H5_FILE_PATH = "dataset/l-line_cable_3/label/001_labeled.hdf5"

# =================================================================================================
# 데이터 구조 분석 함수 (Data Structure Analysis Function)
# =================================================================================================
def analyze_h5_structure(file_path: str):
    """
    HDF5 파일의 전체 계층 구조와 모든 데이터셋의 상세 정보(모양, 타입, 차원)를
    재귀적으로 탐색하여 출력합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 경로에 파일이 존재하지 않습니다.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"===== HDF5 파일 구조 분석: '{os.path.basename(file_path)}' =====")

            # 2. 파일의 최상위 속성(metadata)이 있다면 출력
            if f.attrs:
                print("\n[파일 메타데이터]")
                for key, value in f.attrs.items():
                    print(f"  - {key}: {value}")

            print("\n[데이터 계층 구조]")

            # 3. visititems를 사용하여 모든 객체(그룹, 데이터셋)를 순회
            #    visitor 함수가 각 객체에 대해 호출됩니다.
            f.visititems(visitor)

            print("\n==================== 분석 완료 ====================")

    except Exception as e:
        print(f"파일을 열거나 분석하는 중 오류가 발생했습니다: {e}")


def visitor(name: str, obj: h5py.HLObject):
    """
    h5py의 visititems가 호출하는 콜백 함수입니다.
    객체의 이름(경로)과 객체 자체를 인자로 받습니다.
    """
    # 객체의 깊이(depth)에 따라 들여쓰기 수준을 결정합니다.
    depth = name.count('/')
    indent = "    " * depth

    # 객체가 데이터셋(Dataset)인 경우 상세 정보를 출력합니다.
    if isinstance(obj, h5py.Dataset):
        # shape 튜플을 "dim_0: size, dim_1: size, ..." 형태의 문자열로 변환
        dims = ", ".join([f"dim_{i}: {size}" for i, size in enumerate(obj.shape)])
        
        # 데이터셋 이름, 모양, 차원 정보, 데이터 타입을 출력
        print(f"{indent} L-- 💿 Dataset: '{os.path.basename(name)}'")
        print(f"{indent}     |-- Shape: ({dims})")
        print(f"{indent}     '-- Dtype: {obj.dtype}")

    # 객체가 그룹(Group)인 경우, 그룹 이름을 출력합니다.
    elif isinstance(obj, h5py.Group):
        print(f"{indent} L-- 📁 Group: '{os.path.basename(name)}'")


# =================================================================================================
# 메인 실행 블록 (Main Execution Block)
# =================================================================================================
if __name__ == '__main__':
    analyze_h5_structure(H5_FILE_PATH)
