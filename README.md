# Multimodal Unit Task Annotation Tool

A PyQt-based annotation tool for labeling unit-task segments in multimodal robot datasets.

The tool is designed for research workflows where image streams, numeric time-series, and metadata need to be inspected together before assigning segment-level task labels. It supports HDF5 datasets, ROS 2 MCAP bags, MP4 videos, image-folder sequences, configurable data views, timeline-based labeling, CSV labels, and label-embedded HDF5 exports.

## Features

- Import HDF5 (`.h5`, `.hdf5`), ROS 2 MCAP (`.mcap`), and MP4 files.
- Import all images directly inside a selected folder as one ordered sequence.
- Decode MP4 and image-folder frames lazily to limit memory usage.
- Visualize multiple synchronized data views in a resizable splitter grid.
- Select streams by namespace and modality.
- Display image streams, numeric time-series, and text/metadata streams.
- Pop individual data views out into separate windows and dock them back.
- Label unit-task segments on a shared timeline.
- Import existing label files with or without the original source data loaded.
- Export frame-wise CSV labels or a copied HDF5 file with embedded labels.
- Configure runtime defaults, HDF5 stream mapping, and view grouping with YAML files.

## Repository Layout

```text
.
├── main_app.py              # Application entry point and main window controller
├── main_window.ui           # Qt Designer UI definition
├── app_config.py            # YAML-backed application configuration
├── data_loader.py           # Multiformat dataset loading logic
├── data_models.py           # Shared dataset/session data structures
├── labeling_io.py           # Label import/export helpers
├── media_sources.py         # Lazy MP4 and image-folder frame sources
├── media_label_exporter.py  # Class-organized MP4/image split export
├── progress_dialog.py       # Import/export progress monitoring dialog
├── settings_dialogs.py      # Data-view and YAML configuration editors
├── source_dialog.py         # Unified file or image-folder selector
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

HDF5, MP4, and image-folder workflows can run without ROS 2. MCAP workflows require a ROS 2 environment that provides `rosbag2_py` and the relevant message definitions.

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

1. Click `Import`, then select either one supported file or one image folder.
2. Choose namespaces and streams in each data view.
3. Move through the timeline with the slider or navigation buttons.
4. Start and stop labeling with `Start`, `Space`, or `Enter`.
5. Enter class IDs for each segment.
6. Click `Export` to write the configured label output.

Exported labels are written to a `label/` directory next to the source file or image folder by default. HDF5 sources produce a copied `<source>_labeled.h5` or `.hdf5` file with embedded labels; other sources produce a frame-wise CSV.

The selected file's parent directory becomes the workspace, and `Previous`/`Next` navigates all directly contained supported files regardless of format. Selecting one image imports every image in that directory as one sequence. Selecting a parent directory that contains image-sequence folders makes those folders the `Previous`/`Next` navigation entries.

Before `Previous` or `Next` changes the active source, a confirmation dialog warns that current timeline labels may need to be exported. `Cancel` is the default action.

Import, label import, and export operations display a modal progress window with the active stage, percentage, elapsed time, and estimated remaining time. Split exports update progress while writing video frames or copying labeled images.

The main menu provides the same common commands from the upper-left corner:

- `File`: import a file/folder, import labels, export labels, navigate files, and exit
- `Settings`: edit Data View parameters or directly edit View/App/HDF5/Media Export YAML files

## Configuration

The application is configured through YAML files in `configs/`.

### `configs/app_settings.yaml`

Controls application-level defaults such as:

- Window size
- Supported file extensions
- Default number of data views
- Default image-sequence FPS
- Timeline resolution
- Default zoom window
- Label dataset name
- Default stylesheet path

### `configs/view_config.yaml`

Controls stream grouping and display behavior:

- Namespace order
- Namespace prefix matching
- Default stream selection
- Data View count, plot mode, and zoom window
- Depth colormap and visualization range
- Overlay display fields
- Overlay colors

Use `Settings > Data View Settings...` to apply or save the core parameters. Use `Settings > View Config > Browse...` to switch the active View config during the current session. The YAML editors validate content before saving; application startup defaults take effect after restarting the application.

### `configs/hdf5_mapping.yaml`

Defines how HDF5 groups are interpreted as GUI streams:

- Image stream groups
- Numeric time-series groups
- Timestamp datasets
- Value datasets
- Optional numeric field labels

## Label Format

Non-HDF5 exports use one CSV file with one row per timeline frame:

- `frame_index`: zero-based frame index
- `timestamp_sec`: timestamp mapped to the imported data timeline
- `class_id_1`: primary class ID, or `-1` when unlabeled
- `class_id_2`: optional secondary class ID, or `-1` when unused

CSV label files can be imported with or without the original data loaded. Exported labeled HDF5 files contain both a frame-wise `labels` dataset and a `label_segments` table. Importing one of these files automatically restores its timeline labels without a separate `Import Label` operation.

### Media Label Export Modes

`configs/media_label_export.yaml` controls label and media exports:

- `basic`: export only the frame-wise label CSV for non-HDF5 sources
- `split`: additionally export each labeled segment under its class folder and create root-level `metadata.csv`
- `hdf5.mode: embedded` (default): copy the source HDF5 and embed current labels in the new file
- `hdf5.mode: csv`: retain CSV-only export for HDF5 sources

The embedded HDF5 filename preserves the original extension and uses `hdf5.output_name_suffix` (default `_labeled`). Set `hdf5.export_csv_sidecar` to also create the frame-wise CSV. Class folder names default to `class_<id>`. Set `split.class_names` to use semantic names such as `idle`, `approach`, or `grasp`. Multi-class segments are copied to every corresponding class folder by default. Metadata maps each relative output path to its class, source, and frame range.

## Development

Run a basic syntax check with:

```bash
python -m py_compile main_app.py app_config.py data_loader.py data_models.py labeling_io.py media_sources.py media_label_exporter.py progress_dialog.py settings_dialogs.py source_dialog.py widgets.py multrecog_core.py multrecog_ui.py custom_msg_parser.py
```

When changing the UI, keep `main_window.ui`, runtime setup in `main_app.py`, and stylesheet rules in `configs/*.qss` aligned.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Third-party dependencies are distributed under their own licenses. In particular, PyQt5 is licensed separately by Riverbank Computing; review its license terms when redistributing this project or derived applications.
