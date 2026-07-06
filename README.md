# Multimodal Unit Task Annotation Tool

Multimodal Unit Task Annotation Tool은 로봇 작업 데이터에서 unit task 구간을 시각적으로 확인하고 라벨링하기 위한 PyQt 기반 GUI 도구입니다. HDF5와 ROS2 MCAP 데이터를 불러와 이미지, numeric time-series, metadata stream을 여러 Data View에 배치하고, timeline 위에서 라벨 구간을 생성/수정/내보낼 수 있습니다.

이 프로젝트는 배포용 앱보다는 연구개발 과정에서 빠르게 데이터셋을 확인하고 라벨 품질을 개선하는 목적에 맞춰 구성되어 있습니다.

## 주요 기능

- HDF5(`.h5`, `.hdf5`) 및 ROS2 MCAP(`.mcap`) 입력 지원
- 기본 9개 Data View와 설정 기반 view 개수 변경
- Data View 내부 splitter 기반 동적 크기 조절
- Data View pop-out/pop-in 지원
- namespace 기반 stream 선택 UI
- 이미지, numeric time-series, text/metadata stream 시각화
- timeline 기반 unit task segment 라벨링
- 기존 label HDF5 import 및 segment timeline 표시
- 원본 데이터 없이 label 파일만 import한 timeline 표시
- HDF5 원본 보존형 label export 및 MCAP sidecar label export
- YAML 기반 앱 설정, HDF5 mapping, view namespace 설정

## 프로젝트 구조

```text
.
├── main_app.py              # PyQt 메인 윈도우와 앱 컨트롤러
├── main_window.ui           # Qt Designer UI 정의
├── app_config.py            # configs/app_settings.yaml 로딩 및 전역 설정
├── data_loader.py           # HDF5/MCAP 데이터 로딩
├── data_models.py           # DatasetSession, DataStream 등 공유 데이터 모델
├── labeling_io.py           # label import/export 유틸리티
├── widgets.py               # Data View, plot/image/text widget
├── multrecog_core.py        # 라벨링 상태 및 핵심 로직
├── multrecog_ui.py          # timeline slider, segment edit dialog
├── custom_msg_parser.py     # 로컬 custom ROS2 message fallback parser
├── configs/
│   ├── app_settings.yaml    # 앱 기본값
│   ├── hdf5_mapping.yaml    # HDF5 stream mapping
│   ├── view_config.yaml     # namespace/view 표시 설정
│   └── *.qss                # 스타일시트
├── environment.yml          # preprocessing conda 환경 export
└── doc/                     # 작업 문서
```

`reference/` 폴더는 구현 참고용 외부 프로젝트이므로 git 추적 대상에서 제외합니다.

## 환경 설정

권장 환경은 Ubuntu 22.04, Python 3.10, ROS2 Humble, Miniconda입니다.

```bash
conda env create -f environment.yml
conda activate preprocessing
```

이미 `preprocessing` 환경이 있는 경우에는 다음처럼 갱신할 수 있습니다.

```bash
conda env update -n preprocessing -f environment.yml --prune
conda activate preprocessing
```

MCAP 파일을 읽으려면 ROS2 Humble의 `rosbag2_py`가 필요합니다. 터미널에서 실행할 때는 아래 순서를 권장합니다.

```bash
source /opt/ros/humble/setup.bash
conda activate preprocessing
python main_app.py
```

IDE에서 바로 실행하는 경우 `main_app.py`가 `/opt/ros/humble`의 Python/shared library 경로를 감지해 한 번 재실행하도록 보강되어 있습니다. 그래도 `No module named 'rosbag2_py'`가 발생하면 ROS2 Humble 설치와 `source /opt/ros/humble/setup.bash` 적용 여부를 먼저 확인합니다.

## 실행 방법

```bash
conda activate preprocessing
python main_app.py
```

앱이 실행되면 `Import` 버튼으로 HDF5 또는 MCAP 파일을 선택합니다. 폴더 안에 같은 episode의 파일들이 있으면 지원 확장자 기준으로 함께 탐색합니다.

## 설정 파일

`configs/app_settings.yaml`

- 앱 제목, 시작 창 크기, 지원 확장자, 기본 Data View 개수, timeline 해상도, label dataset 이름을 관리합니다.
- 기본 스타일시트는 `configs/style_light.qss`입니다.

`configs/view_config.yaml`

- stream namespace 순서와 prefix 규칙을 관리합니다.
- Data View의 namespace 선택 방식과 stream grouping에 사용됩니다.
- overlay 표시 항목과 색상도 이 파일에서 관리합니다.

`configs/hdf5_mapping.yaml`

- HDF5 group을 image stream 또는 numeric time-series stream으로 해석하는 mapping입니다.
- `/recog/states` 같은 task recognition stream label도 여기에서 정의합니다.

## 라벨링 흐름

1. `Import`로 원본 HDF5 또는 MCAP 데이터를 불러옵니다.
2. 각 Data View에서 namespace와 stream을 선택합니다.
3. timeline을 이동하며 `Start` 또는 `Space/Enter`로 라벨링 구간을 시작/종료합니다.
4. class ID를 입력하면 segment가 timeline과 Class View에 반영됩니다.
5. `Export`를 누르면 원본 파일 옆 `label/` 폴더에 `*_labeled.hdf5`가 생성됩니다.

HDF5 원본을 export할 때는 원본 group을 복사하고 label dataset을 추가합니다. MCAP 또는 label-only 흐름에서는 source metadata와 `label_segments` dataset을 포함한 sidecar HDF5를 생성합니다.

## Git 설정

이 프로젝트의 원격 저장소는 다음 URL을 사용합니다.

```bash
git remote add origin https://github.com/Peregrine9363/Multimodal-Unit-Task-Annotation-Tool.git
git branch -M main
```

GitHub token은 보안상 remote URL이나 문서에 저장하지 않습니다. push가 필요할 때는 GitHub CLI, credential manager, 또는 일회성 인증 프롬프트를 사용합니다.
