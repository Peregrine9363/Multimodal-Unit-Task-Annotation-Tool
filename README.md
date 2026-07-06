# Multimodal Unit Task Annotation Tool

A PyQt-based annotation tool for labeling unit-task segments in multimodal robot datasets.

The tool is designed for research workflows where image streams, numeric time-series, and metadata need to be inspected together before assigning segment-level task labels. It supports HDF5 datasets and ROS 2 MCAP bags, configurable data views, timeline-based labeling, and HDF5 label import/export.

## Features

- Import HDF5 (`.h5`, `.hdf5`) and ROS 2 MCAP (`.mcap`) files.
- Visualize multiple synchronized data views in a resizable splitter grid.
- Select streams by namespace and modality.
- Display image streams, numeric time-series, and text/metadata streams.
- Pop individual data views out into separate windows and dock them back.
- Label unit-task segments on a shared timeline.
- Import existing label files with or without the original source data loaded.
- Export labels to HDF5 while preserving source metadata when possible.
- Configure runtime defaults, HDF5 stream mapping, and view grouping with YAML files.

## Repository Layout

```text
.
├── main_app.py              # Application entry point and main window controller
├── main_window.ui           # Qt Designer UI definition
├── app_config.py            # YAML-backed application configuration
├── data_loader.py           # HDF5 and MCAP loading logic
├── data_models.py           # Shared dataset/session data structures
├── labeling_io.py           # Label import/export helpers
├── widgets.py               # Data view widgets and visualization components
├── multrecog_core.py        # Core labeling state logic
├── multrecog_ui.py          # Timeline slider and segment editing UI
├── custom_msg_parser.py     # Fallback parser for selected custom ROS messages
├── configs/
│   ├── app_settings.yaml    # Application defaults
│   ├── hdf5_mapping.yaml    # HDF5 stream mapping rules
│   ├── view_config.yaml     # Namespace and view display configuration
│   └── *.qss                # Qt stylesheets
├── environment.yml          # Conda environment export
└── doc/                     # Project notes and planning documents
```

## Requirements

- Ubuntu 22.04 or a compatible Linux environment
- Miniconda or Anaconda
- Python 3.10
- ROS 2 Humble for MCAP loading through `rosbag2_py`

HDF5 workflows can run without ROS 2. MCAP workflows require a ROS 2 environment that provides `rosbag2_py` and the relevant message definitions.

## Installation

Create the Conda environment from the exported environment file:

```bash
conda env create -f environment.yml
conda activate preprocessing
```

If the environment already exists, update it with:

```bash
conda env update -n preprocessing -f environment.yml --prune
conda activate preprocessing
```

For MCAP files, source ROS 2 before launching the app:

```bash
source /opt/ros/humble/setup.bash
conda activate preprocessing
```

## Usage

Start the GUI:

```bash
python main_app.py
```

Typical workflow:

1. Click `Import` and select an HDF5 or MCAP file.
2. Choose namespaces and streams in each data view.
3. Move through the timeline with the slider or navigation buttons.
4. Start and stop labeling with `Start`, `Space`, or `Enter`.
5. Enter class IDs for each segment.
6. Click `Export` to write a labeled HDF5 file.

Exported labels are written to a `label/` directory next to the source file by default.

## Configuration

The application is configured through YAML files in `configs/`.

### `configs/app_settings.yaml`

Controls application-level defaults such as:

- Window size
- Supported file extensions
- Default number of data views
- Timeline resolution
- Default zoom window
- Label dataset name
- Default stylesheet path

### `configs/view_config.yaml`

Controls stream grouping and display behavior:

- Namespace order
- Namespace prefix matching
- Default stream selection
- Overlay display fields
- Overlay colors

### `configs/hdf5_mapping.yaml`

Defines how HDF5 groups are interpreted as GUI streams:

- Image stream groups
- Numeric time-series groups
- Timestamp datasets
- Value datasets
- Optional numeric field labels

## Label Format

The tool supports two HDF5 label representations:

- Frame-wise label datasets named by `default_label_dataset` in `configs/app_settings.yaml`
- Segment tables stored in `label_segments`

For MCAP and label-only workflows, exported sidecar HDF5 files include source metadata and segment timing information when available.

## Development

Run a basic syntax check with:

```bash
python -m py_compile main_app.py app_config.py data_loader.py data_models.py labeling_io.py widgets.py multrecog_core.py multrecog_ui.py custom_msg_parser.py
```

When changing the UI, keep `main_window.ui`, runtime setup in `main_app.py`, and stylesheet rules in `configs/*.qss` aligned.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Third-party dependencies are distributed under their own licenses. In particular, PyQt5 is licensed separately by Riverbank Computing; review its license terms when redistributing this project or derived applications.
