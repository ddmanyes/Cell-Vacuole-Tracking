## Cell Vacuole Tracking

Pipeline for segmenting cells, tracking them over time, and detecting intracellular bubbles/vacuoles from TIFF time-lapse data.

### Features

- Cell segmentation (Cellpose or simple threshold + watershed)
- Optional LapTrack-based cell tracking
- Bubble detection inside each cell (rb+CLAHE thresholding)
- CSV summaries and QC overlay images

### Project Principles

- **Reproducible**: parameters are centralized in `src/pipeline/pipeline.py` and documented.
- **Traceable**: QC overlays and intermediate masks support audit of each frame.
- **Cell-safe**: bubble ownership requires the bubble to be fully inside one cell.

### Workflow Overview

1. Load TIFF time-lapse and normalize frames.
2. Segment cells (Cellpose or threshold + watershed).
3. Optionally track cells over time using LapTrack.
4. Detect bubbles inside each cell (rb+CLAHE thresholding by default).
5. Export CSVs and QC overlays; optionally save intermediate masks.

### Installation

## Installation Instructions (macOS & Windows Compatible)

This project recommends using `uv` for fast, reliable dependency management.

### 1. Install uv

**macOS / Linux / WSL (Recommended)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

Sync the project dependencies from `pyproject.toml`:

```bash
uv sync
```

This command will:

1. Create a virtual environment (`.venv`) if one doesn't exist.
2. Install all required packages into the environment.

### 3. Platform Notes

- **C Compiler**: Some packages (`cellpose`, `scikit-image`) may require build tools.
  - **Windows**: Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  - **macOS**: Install Xcode Command Line Tools (`xcode-select --install`).
- **Permissions**: If you encounter permission errors, try running as Administrator (Windows) or use `sudo` (macOS/Linux) carefully, though `uv` installs to user directories by default.

### 4. Verifying Installation

Verify that the environment is set up correctly and all dependencies manage to load.

#### Import Test

```bash
# Verify key imports
uv run python -c "import cellpose; import laptrack; import skimage; import cv2; print('Imports successful')"
```

#### Dependency Check

```bash
uv pip check
uv pip list
```

- `uv pip check` verifies there are no conflicting dependencies.

### 5. Development (Optional)

Recommended optional tools for development and testing (not in `pyproject.toml` by default):

- `pytest`: For running unit tests (`src/tests`).
- `jupyter`/`ipykernel`: For interactive notebooks.

### Quick Start

1. Install dependencies (recommended via uv)
 - Run `uv sync` to install all dependencies from `pyproject.toml`.
2. Run the pipeline on the default dataset:

```bash
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff"
```

3. Limit frames for fast iteration:

```bash
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --max-frames 10 --skip-tracking
```

4. Save intermediate masks for fast QC reuse:

```bash
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --save-intermediates
```

### Sample Data

The `sample/` folder contains example outputs from a 30-frame test run:

- **CSVs**: tracking positions, cell summaries, bubble statistics
- **QC Images**: tracking overlays, histograms, bubble detection QC
- **Intermediates**: compressed npz file with masks and bubble labels
- **Animations**: single-cell and multi-cell tracking animations

Use sample data to test visualization tools without running the full pipeline:

```bash
# Test overlay from intermediates
uv run src/tests/overlay_from_npz.py --npz "sample/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0

# Test tracking animation
uv run src/tests/track_single_cell_animation.py --tracking-csv sample/tracking_positions.csv \
 --cell-id 0 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" \
 --npz "sample/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz"
```

### Windows Usage

1. Open PowerShell in the project root.
2. (Optional) Activate venv:

```powershell
H:/細胞偵測/.venv/Scripts/Activate.ps1
```

3. Run with `uv` (recommended):

```powershell
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff"
```

4. If PowerShell blocks scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

5. For multi-line commands in PowerShell, use backtick `` ` `` (not `\`):

```powershell
uv run src/tests/track_single_cell_animation.py `
    --tracking-csv results/tracking_positions.csv `
    --cell-id 0 `
    --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff"
```

### Output Files

Results are written to `results/`:

- `results/<input_stem>.csv`: per-frame, per-cell bubble stats (includes `bubble_area_method`)
- `results/summary.csv`: single-line summary (includes `bubble_area_method_*_rows`)
- `results/cell_summary.csv`: per-cell aggregate stats
- `results/qc/<input_stem>_qc.png`: overlay QC image
- `results/intermediates/<input_stem>_intermediate.npz`: saved masks/bubble labels

Tracking diagnostics (when tracking enabled):

- `results/tracking_summary.csv`
- `results/tracking_positions.csv`
- `results/tracking_lengths.csv`
- `results/tracking_steps.csv`
- `results/tracking_lengths_hist.png`
- `results/tracking_steps_hist.png`
- `results/tracking_overlay.png`

### Key Configuration

Configuration is managed via `config/pipeline_params.yaml`. You can modify this file to adjust segmentation, bubble detection, and output settings without changing the code.

**Key Sections:**

- **segmentation**: Parameters for `cellpose` or `threshold` methods.
- **cellpose**: Model type (`cyto3`), diameter, and flow threshold.
- **bubble**: Parameters for `rb_clahe` (rolling ball + CLAHE) detection.
  - `thresh`, `min_area`, `min_circularity`: Main filters for bubbles.
  - `rb_radius`, `clahe_clip`: Preprocessing settings.
- **output**: Paths for results and QC images.

**Example `config/pipeline_params.yaml` snippet:**

```yaml
segmentation:
  method: cellpose
  min_cell_area: 200

cellpose:
  model_type: cyto3
  diameter: 100

bubble:
  method: rb_clahe
  rb_clahe:
    thresh: 0.28
    min_area: 20
```

To run with default configuration:

```bash
uv run src/pipeline/pipeline.py
```

The script will automatically load `config/pipeline_params.yaml` if it exists.

### Test Utilities

- `src/tests/bubble_frame_preproc_test.py`: single-frame bubble preprocessing comparison
- `src/tests/bubble_single_cell_test.py`: single-cell bubble method sweeps
- `src/tests/bubble_param_sweep.py`: automated parameter sweep for bubble detection (batch testing with metrics)
- `src/tests/cellpose_frame_test.py`: cellpose single-frame QC
- `src/tests/param_sweep.py`: segmentation parameter sweep
- `src/tests/track_single_cell.py`: plot trajectory for a tracked cell
- `src/tests/overlay_from_npz.py`: draw overlays from saved intermediates
- `src/tests/track_single_cell_animation.py`: tracking animation (single/multiple/all cells)
- `src/legacy/bubble_qc_sweep.py`: legacy bubble QC sweep (blob_log)

Example:

```bash
# Single-frame preprocessing comparison
uv run src/tests/bubble_frame_preproc_test.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0

# Automated parameter sweep with metrics CSV
uv run src/tests/bubble_param_sweep.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

Track example:

```bash
uv run src/tests/track_single_cell.py --tracking-csv results/tracking_positions.csv --cell-id 12
```

Overlay from intermediates:

```bash
uv run src/tests/overlay_from_npz.py --npz "results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

Animation examples:

```bash
# Single cell animation
uv run src/tests/track_single_cell_animation.py --tracking-csv results/tracking_positions.csv \
 --cell-id 12 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" \
 --npz "results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --output results/tracking_qc/tracked_12.gif

# Multiple cells by ID
uv run src/tests/track_single_cell_animation.py --tracking-csv results/tracking_positions.csv \
 --cell-ids 0,1,2 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" \
 --npz "results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --output results/tracking_qc/tracked_multi.gif

# Top 5 longest tracks
uv run src/tests/track_single_cell_animation.py --tracking-csv results/tracking_positions.csv \
 --top-n 5 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" \
 --npz "results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --output results/tracking_qc/tracked_top5.gif

# All cells with minimum track length filter
uv run src/tests/track_single_cell_animation.py --tracking-csv results/tracking_positions.csv \
 --all --min-length 10 --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" \
 --npz "results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz" \
 --output results/tracking_qc/tracked_all.gif
```

**Animation Options:**

- `--cell-id N`: Animate single cell with ID N
- `--cell-ids A,B,C`: Animate multiple cells (comma-separated IDs)
- `--top-n N`: Animate top N longest tracks
- `--all`: Animate all tracked cells
- `--min-length N`: Filter tracks shorter than N frames (default: 1)
- `--no-outlines`: Skip cell/bubble outlines for faster rendering
- `--fps N`: Animation speed in frames per second (default: 6)

### Documentation

- `docs/performance_optimization.md`: performance review and optimization notes
- `docs/testing.md`: test workflow and scripts
- `docs/test_log.md`: experiment log template

### Dependencies & Sources

- Cellpose: <https://github.com/MouseLand/cellpose>
- LapTrack: <https://github.com/Noneq/laptrack>
- scikit-image: <https://scikit-image.org/>
- tifffile: <https://github.com/cgohlke/tifffile>
- NumPy: <https://numpy.org/>
- pandas: <https://pandas.pydata.org/>
- tqdm: <https://github.com/tqdm/tqdm>
- matplotlib: <https://matplotlib.org/>
- Pillow: <https://python-pillow.org/>

### Citation Formats

Use one of the following formats when citing this project.

**APA**
Chan, C. R. (2026). Cell Vacuole Tracking (Version 0.1.0) [Computer software].

**BibTeX**

```bibtex
@software{chan2026cellvacuole,
 title = {Cell Vacuole Tracking},
 author = {Chan, Chi Ru},
 year = {2026},
 version = {0.1.0},
 note = {Git repository},
}
```

### Project Structure

- `src/pipeline/`: main pipeline
- `src/tests/`: test and sweep scripts
- `src/legacy/`: deprecated or legacy experiments
- `data/`: input TIFF datasets
- `results/`: generated CSVs and QC images
- `sample/`: example outputs from 30-frame test run
- `docs/`: documentation and experiment notes
- `pyproject.toml`: dependencies and project metadata

### Notes

- Cellpose is required when `SEGMENTATION_METHOD` is set to `"cellpose"`.
- The default input path assumes the `data/` directory structure in this repo.
- `bubble_area_method` indicates whether bubble area is computed from pixel count or blob sigma.

### Sharing & Publication

- Include a copy of the MIT license when sharing code.
- Cite upstream libraries listed above in your publication or README.
- Avoid distributing raw data without permission; share derived metrics and QC figures instead.
- Record parameter settings and dataset identifiers to ensure reproducibility.

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
