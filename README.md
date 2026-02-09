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

This project recommends using a Python virtual environment to ensure dependency consistency and system security. The following steps apply to both macOS and Windows, with notes on common platform differences.

### 1. Python Version Requirements

- **Recommended versions**: Python 3.9 ~ 3.11
- Do not use Python 3.12 (some scientific packages are not yet fully supported)

### 2. Creating a Virtual Environment

#### Using uv (recommended, fast and automatically handles dependencies)

```bash
# Install uv (if not already installed)
pip install uv

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

#### Or using venv + pip

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Installing Dependencies

#### Using uv (recommended - handles everything automatically)

```bash
# Sync dependencies from pyproject.toml (creates/activates venv if needed)
uv sync

# Or if you prefer manual venv creation:
uv venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
uv sync
```

#### Alternative: Using pip

```bash
pip install -r requirements.txt
# or
pip install .
```

### 4. Platform Differences and Notes

- **C Compiler Requirements**  
  - Some packages (such as `scikit-image`, `cellpose`, `cython`) require a C/C++ compiler.
  - **Windows**: It is recommended to install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  - **macOS**: Install Xcode Command Line Tools (run `xcode-select --install`).

- **Pillow, tifffile**  
  - Require system support for libjpeg, zlib, etc. On macOS, you can install via Homebrew; on Windows, it is usually handled automatically.

- **cellpose/scikit-image**  
  - These packages depend on numpy, scipy, cython. Ensure the installation order is correct (uv/pip will handle this automatically).

- **Permission Issues**  
  - On macOS, if you encounter permission issues during installation, add `--user` or use a virtual environment.
  - On Windows, run the command prompt as administrator if you encounter permission errors.

### 5. Dependency Integrity Check and Suggested Additions

#### Required Scientific Packages (confirm they are included in requirements/pyproject):

- numpy
- scipy
- pandas
- matplotlib
- scikit-image
- tifffile
- pillow
- opencv-python
- cellpose
- networkx
- cython

#### Suggested Additions (consider adding if not listed in dependencies):

- **scipy**: Commonly used for image processing and numerical operations
- **networkx**: For cell tracking/graph analysis
- **cython**: To accelerate some scientific packages
- **opencv-python**: For image processing
- **pytest**: For unit testing
- **jupyter**: For interactive analysis
- **tqdm**: For progress bars
- **h5py**: If HDF5 format handling is needed
- **pyyaml**: If YAML configuration parsing is needed

> If there are omissions, please add them to `pyproject.toml` or `requirements.txt`.

### 6. Verifying Installation Success

#### Import Test

```bash
# If using uv
uv run python -c "import numpy; import scipy; import pandas; import matplotlib; import skimage; import tifffile; import PIL; import cv2; import cellpose; import networkx; import cython"

# Or activate venv and run
python -c "import numpy; import scipy; import pandas; import matplotlib; import skimage; import tifffile; import PIL; import cv2; import cellpose; import networkx; import cython"
```
- If no errors, installation is successful.

#### Dependency Check

```bash
# Using uv
uv pip check
uv pip list

# Or using pip
pip check
pip list
```
- Check for missing or conflicting packages.

---

If you encounter installation issues, please provide the error message and your operating system version for assistance.

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
Configuration is defined in `src/pipeline/pipeline.py`:
- `SEGMENTATION_METHOD`: `"cellpose"` or `"threshold"`
- `BUBBLE_METHOD`: `"rb_clahe"` (current default)
- `BUBBLE_TH_THRESH`, `BUBBLE_TH_MIN_AREA`, `BUBBLE_TH_MIN_CIRCULARITY`
- `BUBBLE_TH_QC_*`: QC-only parameters for `rb_clahe` overlays

### Test Utilities
- `src/tests/bubble_frame_preproc_test.py`: single-frame bubble preprocessing comparison
- `src/tests/bubble_single_cell_test.py`: single-cell bubble method sweeps
- `src/tests/cellpose_frame_test.py`: cellpose single-frame QC
- `src/tests/param_sweep.py`: segmentation parameter sweep
- `src/tests/track_single_cell.py`: plot trajectory for a tracked cell
- `src/tests/overlay_from_npz.py`: draw overlays from saved intermediates
- `src/tests/track_single_cell_animation.py`: tracking animation (single/multiple/all cells)
- `src/legacy/bubble_qc_sweep.py`: legacy bubble QC sweep (blob_log)

Example:
```bash
uv run src/tests/bubble_frame_preproc_test.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
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
- Cellpose: https://github.com/MouseLand/cellpose
- LapTrack: https://github.com/Noneq/laptrack
- scikit-image: https://scikit-image.org/
- tifffile: https://github.com/cgohlke/tifffile
- NumPy: https://numpy.org/
- pandas: https://pandas.pydata.org/
- tqdm: https://github.com/tqdm/tqdm
- matplotlib: https://matplotlib.org/
- Pillow: https://python-pillow.org/

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
