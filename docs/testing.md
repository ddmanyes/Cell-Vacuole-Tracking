# Testing Guide

This guide provides a systematic approach to testing and optimizing the cell vacuole tracking pipeline, from initial TIFF data to final bubble detection parameters.

## Overview

The testing workflow follows these steps:

1. **Cell Segmentation Testing** - Validate Cellpose parameters for accurate cell detection
2. **Bubble Detection Testing** - Optimize bubble detection parameters within cells
3. **Full Pipeline Validation** - Run complete analysis with chosen parameters
4. **Result Analysis** - Evaluate outputs and iterate as needed

## Step 1: Cell Segmentation Testing (Cellpose)

### Purpose

Test Cellpose segmentation parameters to ensure cells are accurately detected before proceeding to bubble analysis.

### How to Run

Use the cellpose frame test script:

```bash
uv run src/tests/cellpose_frame_test.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

### Adjustable Parameters

#### Cellpose Model Parameters

- **model_type**: `"cyto"` (cytoplasm) or `"nuclei"` (nuclei) - use `"cyto"` for cell segmentation
- **diameter**: Expected cell diameter in pixels (default: 100)
  - Measure a few cells in your image to estimate
  - Too small: cells split; Too large: cells merge
- **cellprob_threshold**: Cell probability threshold (default: 0.6)
  - Lower values: more sensitive detection, may include noise
  - Higher values: more conservative, may miss faint cells
- **flow_threshold**: Flow field threshold (default: 0.4)
  - Affects boundary detection precision

#### Preprocessing Parameters

- **normalize**: Whether to normalize image intensity (default: True)
- **sharpen**: Apply sharpening filter (default: False)
- **smooth**: Apply smoothing filter (default: False)

### How to Evaluate Results

1. **Visual Inspection**: Check overlay images for:
   - All cells detected (no missing cells)
   - No over-segmentation (cells split incorrectly)
   - No under-segmentation (adjacent cells merged)
   - Cell boundaries align with actual cell edges

2. **Quantitative Metrics**:
   - Cell count should be reasonable for your sample
   - Cell sizes should be consistent
   - Compare with manual counting if possible

### Output Files

- `results/cellpose/cellpose_frame_<n>_d<diameter>.png`: Segmentation overlay
- `results/cellpose/cellpose_frame_<n>_d<diameter>_masks.png`: Binary masks
- `results/cellpose/cellpose_frame_<n>_d<diameter>_flows.png`: Flow fields

### Parameter Optimization Tips

1. Start with default diameter=100, adjust based on your cells
2. If cells are touching, increase flow_threshold
3. If detecting too much background noise, increase cellprob_threshold
4. Test on multiple frames to ensure consistency

## Step 2: Cell Segmentation Parameter Sweep

### Purpose

Test multiple cell segmentation parameter combinations to find the optimal settings for your data. This is useful when Cellpose results are not ideal or you want to use traditional segmentation methods.

### How to Run

```bash
uv run src/tests/param_sweep.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

### Customizing Parameters

**Recommended**: Edit `config/pipeline_params.yaml` to customize parameter sweeps:

```yaml
segmentation_sweep:
  baseline:
    gaussian_sigma: 1.0          # 高斯平滑的標準差
    min_cell_area: 200           # 最小細胞面積
    peak_min_distance: 7         # 細胞中心最小間距
    # ... more parameters
  
  variants:
    - name: clahe_only
      params:
        use_clahe: true
    
    - name: custom_test          # Add your own variants!
      params:
        peak_min_distance: 12
        use_clahe: true
```

**Key Parameters and Their Effects**:

- **`peak_min_distance`**: Controls cell splitting granularity
  - Too small → over-segmentation (one cell split into multiple)
  - Too large → under-segmentation (multiple cells merged)
  - Recommended: cell diameter / 10 to / 5 (e.g., 5-20)

- **`min_cell_area`**: Filters noise by removing small objects
  - Adjust based on minimum expected cell size (50-500 pixels)

- **`closing_disk`**: Connects fragmented cell boundaries
  - Larger values → stronger connection of gaps
  - Recommended: 2-7

- **`use_clahe`**: Enhances local contrast
  - Use when image has low contrast or uneven illumination

- **`bg_subtract`**: Background removal method
  - `"gaussian"`: Smooth background estimation
  - `"rolling_ball"`: Better for cellular images
  - `rb_radius` should be ~half of cell diameter

### Output Files

- `results/variants/variant_<name>.png`: QC overlays for each parameter set
- `results/variants/variant_metrics.csv`: Quantitative comparison table

### How to Evaluate Results

1. **Check `variant_metrics.csv`**:
   - `label_count`: Should match expected cell count
   - `coverage`: Typically 0.3-0.7 is reasonable
   - `inside_outside_contrast`: Higher = better cell/background separation

2. **Visually inspect PNG overlays**:
   - Cell boundaries should align with actual cells
   - No over-segmentation or under-segmentation
   - No excessive background noise

3. **Common Problems and Solutions**:
   - **Over-segmentation** → Increase `peak_min_distance`
   - **Under-segmentation** → Decrease `peak_min_distance`
   - **Too much noise** → Increase `min_cell_area` or enable `bg_subtract`
   - **Incomplete boundaries** → Increase `closing_disk`
   - **Uneven illumination** → Use `adaptive_threshold` variant

4. **Update pipeline parameters**:
   - Once you find optimal settings, update them in `config/pipeline_params.yaml` under `cellpose` or `segmentation` section

## Step 3: Bubble Detection Testing

### Purpose

Test bubble detection parameters within segmented cells to optimize vacuole identification.

### How to Run

#### Single Frame Preprocessing Comparison

```bash
uv run src/tests/bubble_frame_preproc_test.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

#### Single Cell Parameter Sweeps

```bash
uv run src/tests/bubble_single_cell_test.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0 --cell-id 0
```

#### Automated Parameter Sweep (Recommended)

**Purpose**: Systematically test multiple parameter combinations on a single frame to find optimal bubble detection settings. This script automates the testing process and generates quantitative metrics for comparison.

**How to Run**:

```bash
uv run src/tests/bubble_param_sweep.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --frame 0
```

**Parameter Sweeps Included**:
The script automatically tests multiple parameter sets including:

- **Baseline**: Uses parameters from `config/pipeline_params.yaml`
- **Threshold sweep**: Tests multiple threshold values (default: 0.25, 0.30, 0.35, 0.40)
- **Min area sweep**: Tests different minimum bubble sizes (default: 5, 10, 15, 20 pixels)
- **CLAHE clip sweep**: Tests various contrast enhancement levels (default: 0.04, 0.06, 0.08, 0.10, 0.15)
- **Recommended presets**: `sensitive_small` (for small bubbles) and `balanced` (general purpose)

**Sweep Ranges Customization**:
You can customize the parameter ranges in `config/pipeline_params.yaml`:

```yaml
bubble_sweep:
  thresh: [0.25, 0.30, 0.35, 0.40]
  min_area: [5, 10, 15, 20]
  clahe_clip: [0.04, 0.06, 0.08, 0.10, 0.15]
```

**Output Files**:

- `results/bubble_param_sweep/<variant_name>.png`: QC overlay for each parameter combination
- `results/bubble_param_sweep/bubble_param_metrics.csv`: Quantitative metrics table with:
  - `total_bubbles`: Total number of bubbles detected
  - `total_bubble_area`: Total area of all bubbles
  - `cells_with_bubbles`: Number of cells containing at least one bubble
  - `total_cells`: Total number of cells in the frame
  - `avg_bubbles_per_cell`: Average bubbles per cell
  - `avg_bubble_area`: Average bubble size

**How to Evaluate Results**:

1. Review the metrics CSV to identify parameter sets with reasonable bubble counts
2. Visually inspect the corresponding PNG overlays to verify accuracy
3. Balance sensitivity (detecting all real bubbles) with specificity (avoiding false positives)
4. Consider biological context: expected bubble sizes and frequency

**Optimization Tips**:

- Start with the baseline and compare against sweep variants
- Look for parameter combinations that give stable results across multiple frames
- Use `avg_bubbles_per_cell` as a quick quality metric
- Check `avg_bubble_area` to ensure detected bubbles are biologically plausible
- The script prints top 5 variants by total bubbles detected for quick reference

### Adjustable Parameters

#### Preprocessing Methods

- **rb_clahe**: Rolling ball background subtraction + CLAHE enhancement (recommended)
- **dog_clahe**: Difference of Gaussians + CLAHE enhancement

#### rb_clahe Parameters

- **rb_radius**: Rolling ball radius for background subtraction (default: 50)
  - Larger values: remove larger background structures
  - Smaller values: preserve more local variations
- **clahe_clip_limit**: CLAHE contrast enhancement limit (default: 0.02)
  - Higher values: more aggressive enhancement
  - Lower values: subtler enhancement
- **clahe_grid_size**: CLAHE tile grid size (default: 8)
  - Smaller grids: more local adaptation
  - Larger grids: more uniform enhancement

#### Thresholding Parameters

- **threshold**: Binary threshold value (default: 0.15)
  - Lower values: detect more bubbles (may include noise)
  - Higher values: detect fewer, more confident bubbles
- **min_area**: Minimum bubble area in pixels (default: 10)
  - Filter out small noise particles
- **min_circularity**: Minimum circularity (0-1) (default: 0.3)
  - 1.0 = perfect circle, lower values allow more irregular shapes

#### Morphological Parameters

- **morph_opening**: Apply morphological opening (default: True)
- **morph_closing**: Apply morphological closing (default: True)
- **morph_disk_size**: Morphological operation disk size (default: 2)

### How to Evaluate Results

1. **Visual Inspection**:
   - Bubbles should be inside cell boundaries
   - No bubbles detected outside cells
   - Bubble shapes should be reasonably circular
   - No obvious missed bubbles or false positives

2. **Quantitative Metrics**:
   - Bubble count per cell should be biologically reasonable
   - Bubble sizes should be consistent within cells
   - Compare bubble_area_method in output CSVs

3. **Parameter Sweep Analysis**:
   - Use generated PNG grids to see parameter effects
   - Look for parameter ranges that give consistent results
   - Balance sensitivity (detecting all bubbles) vs specificity (avoiding noise)

### Output Files

#### bubble_frame_preproc_test.py

- `results/bubble_preproc_frame/frame_<n>_compare.png`: Side-by-side comparison of methods
- `results/bubble_preproc_frame/frame_<n>_rb_clahe_stats.csv`: Per-cell bubble statistics
- `results/bubble_preproc_frame/frame_<n>_dog_clahe_stats.csv`: Per-cell bubble statistics

#### bubble_single_cell_test.py

- `results/bubble_single_cell/cell_<id>_frame_<n>.png`: Single cell analysis
- `results/bubble_single_cell/cell_<id>_frame_<n>_clahe_sweep.png`: CLAHE parameter grid
- `results/bubble_single_cell/cell_<id>_frame_<n>_threshold_sweep.png`: Threshold parameter grid
- Various debug images for parameter optimization

### Parameter Optimization Tips

1. **Start with rb_clahe**: Generally more robust than dog_clahe
2. **Adjust threshold first**: Find the sweet spot where bubbles are detected but noise is minimal
3. **Tune CLAHE**: If bubbles are hard to see, increase clip_limit; if over-enhanced, decrease it
4. **Set morphological filters**: Use opening to remove small noise, closing to fill gaps
5. **Validate on multiple cells**: Parameters should work across different cell types

## Step 3: Full Pipeline Testing

### Running with Optimized Parameters

Once you have good parameters, update them in `config/pipeline_params.yaml`:

```yaml
cellpose:
  model_type: cyto3
  diameter: 100
  cellprob_threshold: 0.6
  flow_threshold: 0.4

bubble:
  method: rb_clahe
  rb_clahe:
    thresh: 0.15
    min_area: 10
    min_circularity: 0.3
    rb_radius: 50
    clahe_clip: 0.02
```

### Test Run Commands

```bash
# Quick test with limited frames
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --max-frames 5 --save-intermediates

# Full test with tracking
uv run src/pipeline/pipeline.py --input "data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff" --max-frames 30 --save-intermediates
```

### Validation Outputs

#### CSV Files

- `results/<input_stem>.csv`: Per-frame, per-cell bubble data
- `results/summary.csv`: Overall statistics
- `results/cell_summary.csv`: Per-cell aggregates

#### QC Images

- `results/qc/<input_stem>_qc.png`: Bubble detection overlay
- `results/tracking_overlay.png`: Cell tracking visualization

#### Intermediate Files

- `results/intermediates/<input_stem>_intermediate.npz`: Saved masks and bubbles for fast QC

## Recommended Workflow

1. **Cell Segmentation**:
   - Run `cellpose_frame_test.py` on frame 0
   - Adjust diameter, cellprob_threshold, flow_threshold
   - Verify cell boundaries are accurate

2. **Bubble Detection**:
   - Run `bubble_single_cell_test.py` on a representative cell
   - Adjust threshold, clahe parameters, morphological filters
   - Run `bubble_frame_preproc_test.py` to compare methods

3. **Parameter Integration**:
   - Update parameters in `pipeline.py`
   - Run small test (5-10 frames) to validate

4. **Full Validation**:
   - Run complete pipeline with tracking
   - Check summary statistics and QC images
   - Iterate parameters if needed

5. **Documentation**:
   - Record final parameters in `docs/test_log.md`
   - Save parameter sets for reproducibility

## Troubleshooting

### Common Issues

- **No cells detected**: Lower cellprob_threshold or check image contrast
- **Cells merging**: Increase flow_threshold or diameter
- **Too many bubbles**: Increase threshold or min_area
- **Missing bubbles**: Decrease threshold or clahe_clip_limit
- **Bubbles outside cells**: Check cell segmentation masks

### Performance Tips

- Use `--save-intermediates` to avoid recomputing segmentation
- Test on single frames first, then scale up
- Use sample data for quick parameter testing
- Document parameter changes in test_log.md

## Reference

- Cellpose documentation: <https://cellpose.readthedocs.io/>
- scikit-image morphology: <https://scikit-image.org/docs/stable/api/skimage.morphology.html>
- CLAHE: <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>
