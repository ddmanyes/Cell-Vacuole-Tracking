# Implementation Plan

## Scope
Use TIFF time-lapse images to build a reproducible pipeline for cell segmentation, tracking, and intracellular bubble analysis with CSV and QC outputs.

## Inputs
- data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff
- data/bafA1/Group 2_wellA2_RI_MIP_stitched.tiff
- data/control/Group 1_wellA1_RI_MIP_stitched.tiff
- data/control/Group 2_wellA2_RI_MIP_stitched.tiff

## Steps
1. Verify TIFF metadata (shape, dtype, intensity range) and report summary.
2. Pre-check dependencies by listing them in pyproject.toml and running `uv lock --dry-run` (cellpose, laptrack, scikit-image, tifffile, numpy, pandas, tqdm, matplotlib).
3. Install dependencies after dry-run resolves cleanly.
4. Implement a batch script to read TIFFs, run Cellpose, track cells, and detect bubbles.
5. Generate CSV outputs and QC overlays for a small sample frame per file.
6. Document usage and parameters in README and a short QC note.

## Outputs
- results/*.csv
- results/qc/*.png
- README updates for running the pipeline

## Progress
- [x] Step 1: Verify TIFF metadata (shape, dtype, intensity range).
- [x] Step 2: Pre-check dependencies with `uv lock --dry-run`.
- [x] Step 3: Install dependencies and generate uv.lock.
- [x] Step 4: Implement batch pipeline script (Added YAML config support).
- [/] Step 5: Generate CSV outputs and QC overlays (Running verification script).
- [x] Step 6: Document usage and QC notes (Metadata detailed in pyproject.toml).

## Recent Updates
- Implemented YAML configuration loading in `src/pipeline/pipeline.py`.
- Updated `.gitignore` to allow tracking of sample CSV files.
- Refined `pyproject.toml` with author metadata and `pytest` dev dependency.
