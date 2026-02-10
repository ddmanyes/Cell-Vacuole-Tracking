"""
Segmentation parameter sweep for visual QC and metrics.

This script automates the testing of multiple cell segmentation parameter combinations
on a single frame, generating both visual overlays and quantitative metrics for comparison.

Features:
- Automatically loads parameter variants from config/pipeline_params.yaml
- Tests systematic sweeps across preprocessing, thresholding, and watershed parameters
- Generates QC overlays for visual inspection (boundaries + colored masks)
- Exports CSV metrics for quantitative comparison

Configuration:
Parameter variants are defined in config/pipeline_params.yaml under 'segmentation_sweep':
    segmentation_sweep:
      baseline:
        gaussian_sigma: 1.0
        min_cell_area: 200
        peak_min_distance: 7
        # ... other default parameters
      variants:
        - name: clahe_only
          description: "僅使用 CLAHE 對比增強"
          params:
            use_clahe: true
        - name: bg_rolling_ball_only
          params:
            bg_subtract: rolling_ball
        # ... more variants

Usage:
    # Run sweep on frame 1 (default)
    uv run src/tests/param_sweep.py --input <tiff_path> --frame 1
    
    # Custom output directory
    uv run src/tests/param_sweep.py --output-dir results/custom_variants
    
    # Test different frames to verify consistency
    uv run src/tests/param_sweep.py --frame 0
    uv run src/tests/param_sweep.py --frame 5

Output Files:
- results/variants/variant_<name>.png: QC overlays (boundaries + colored mask)
- results/variants/variant_metrics.csv: Quantitative metrics for all variants

Workflow:
1. Configure parameter variants in config/pipeline_params.yaml
2. Run the sweep on a representative frame
3. Review variant_metrics.csv (focus on label_count and coverage)
4. Visually inspect corresponding PNG overlays
5. Update config/pipeline_params.yaml with chosen optimal parameters
6. Run full pipeline with optimized settings

Tips:
- Check label_count against expected cell count
- Coverage 0.3-0.7 is typically reasonable
- High inside_outside_contrast indicates good cell/background separation
- Compare multiple frames to ensure parameter robustness
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, threshold_otsu, threshold_local, sobel
from skimage.feature import peak_local_max
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.measure import label as label_img
from skimage.segmentation import watershed, find_boundaries
from skimage.restoration import rolling_ball
from scipy import ndimage

import yaml

# Load config if available
CONFIG_PATH = Path("config/pipeline_params.yaml")
CONFIG = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

DEFAULT_INPUT = Path("data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff")
if CONFIG.get("input", {}).get("tiff_path"):
    DEFAULT_INPUT = Path(CONFIG["input"]["tiff_path"])

DEFAULT_OUTDIR = Path("results/variants")


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        frame = (frame - min_val) / (max_val - min_val)
    else:
        frame = np.zeros_like(frame)
    return frame


def preprocess(frame, use_clahe=False, bg_subtract=None, bg_sigma=25, rb_radius=50):
    img = normalize_frame(frame)

    if bg_subtract == "gaussian":
        background = gaussian(img, sigma=bg_sigma)
        img = img - background
        img = normalize_frame(img)
    elif bg_subtract == "rolling_ball":
        background = rolling_ball(img, radius=rb_radius)
        img = img - background
        img = normalize_frame(img)

    if use_clahe:
        img = equalize_adapthist(img, clip_limit=0.02)
        img = normalize_frame(img)

    return img


def segment_variant(
    frame,
    *,
    use_clahe=False,
    bg_subtract=None,
    bg_sigma=25,
    rb_radius=50,
    use_local_thresh=False,
    local_block_size=51,
    local_offset=-0.01,
    gaussian_sigma=1.0,
    min_cell_area=200,
    closing_disk=3,
    remove_holes_area=150,
    peak_min_distance=7,
    peak_threshold_abs=None,
    use_sobel_ws=False,
):
    img = preprocess(
        frame,
        use_clahe=use_clahe,
        bg_subtract=bg_subtract,
        bg_sigma=bg_sigma,
        rb_radius=rb_radius,
    )

    smoothed = gaussian(img, sigma=gaussian_sigma)

    if use_local_thresh:
        thresh = threshold_local(smoothed, block_size=local_block_size, offset=local_offset)
        binary = smoothed > thresh
    else:
        try:
            thresh = threshold_otsu(smoothed)
        except ValueError:
            thresh = float(np.mean(smoothed))
        binary = smoothed > thresh

    binary = remove_small_objects(binary, min_size=min_cell_area)
    binary = remove_small_holes(binary, area_threshold=remove_holes_area)
    binary = binary_closing(binary, disk(closing_disk))

    distance = ndimage.distance_transform_edt(binary)
    peaks = peak_local_max(
        distance,
        min_distance=peak_min_distance,
        threshold_abs=peak_threshold_abs,
        labels=binary,
    )

    markers = np.zeros_like(smoothed, dtype=np.int32)
    for idx, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = idx

    if markers.max() == 0:
        labels = label_img(binary)
    else:
        if use_sobel_ws:
            gradient = sobel(smoothed)
            labels = watershed(gradient, markers, mask=binary)
        else:
            labels = watershed(-distance, markers, mask=binary)

    return img, labels


def qc_overlay(img, mask, title, brightness=1.25, gamma=0.9):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Brightness/gamma adjustment for clearer visualization
    vis = np.clip(img * brightness, 0, 1)
    vis = np.power(vis, gamma)

    ax = axes[0]
    ax.imshow(vis, cmap="gray")
    boundaries = find_boundaries(mask, mode="outer")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = boundaries.astype(np.float32)
    ax.imshow(overlay)
    ax.set_title(f"{title}: boundaries")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(vis, cmap="gray", alpha=0.5)
    mask_rgb = np.zeros((*mask.shape, 3))
    unique_labels = np.unique(mask)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label_id in enumerate(unique_labels):
        if label_id == 0:
            continue
        mask_rgb[mask == label_id] = colors[i % len(colors)][:3]
    ax.imshow(mask_rgb)
    ax.set_title(f"{title}: mask")
    ax.axis("off")

    plt.tight_layout()
    return fig


def compute_metrics(img, mask):
    total_px = mask.size
    mask_px = int(np.sum(mask > 0))
    coverage = mask_px / total_px if total_px else 0.0

    boundaries = find_boundaries(mask, mode="outer")
    boundary_ratio = float(np.sum(boundaries)) / total_px if total_px else 0.0

    inside_mean = float(np.mean(img[mask > 0])) if mask_px else 0.0
    outside_mean = float(np.mean(img[mask == 0])) if mask_px < total_px else 0.0
    contrast = inside_mean - outside_mean

    labels = np.unique(mask)
    label_count = int(len(labels) - (1 if 0 in labels else 0))

    return {
        "coverage": coverage,
        "boundary_ratio": boundary_ratio,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_outside_contrast": contrast,
        "label_count": label_count,
        "avg_area": float(mask_px / label_count) if label_count else 0.0,
    }


def main(input_path, frame_index, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]

    # Load variants from config or use hardcoded defaults
    sweep_conf = CONFIG.get("segmentation_sweep", {})
    baseline_params = sweep_conf.get("baseline", {})
    config_variants = sweep_conf.get("variants", [])
    
    if config_variants:
        # Use variants from config file
        print(f"Loading {len(config_variants)} variants from config/pipeline_params.yaml")
        variants = []
        for var in config_variants:
            # Merge baseline params with variant-specific params
            merged_params = {**baseline_params, **var.get("params", {})}
            variants.append({
                "name": var["name"],
                "params": merged_params
            })
    else:
        # Fallback to hardcoded defaults for backward compatibility
        print("No config found, using hardcoded default variants")
        variants = []

        # Baseline for comparison
        variants.append({"name": "base_otsu", "params": dict()})

        # 1) Pre-processing
        variants.append({
            "name": "clahe_only",
            "params": dict(use_clahe=True),
        })
        variants.append({
            "name": "bg_gaussian_only",
            "params": dict(bg_subtract="gaussian", bg_sigma=25),
        })
        variants.append({
            "name": "bg_rolling_ball_only",
            "params": dict(bg_subtract="rolling_ball", rb_radius=50),
        })
        variants.append({
            "name": "clahe_bg_gaussian",
            "params": dict(use_clahe=True, bg_subtract="gaussian", bg_sigma=25),
        })
        variants.append({
            "name": "clahe_bg_rolling_ball",
            "params": dict(use_clahe=True, bg_subtract="rolling_ball", rb_radius=50),
        })

        # 2) Marker control
        variants.append({
            "name": "marker_min_distance_14",
            "params": dict(peak_min_distance=14),
        })
        variants.append({
            "name": "marker_threshold_abs_0p15",
            "params": dict(peak_threshold_abs=0.15),
        })

        # 3) Morphology
        variants.append({
            "name": "morph_closing7_holes300",
            "params": dict(closing_disk=7, remove_holes_area=300),
        })

        # 4) Adaptive threshold
        variants.append({
            "name": "adaptive_threshold",
            "params": dict(use_local_thresh=True, local_block_size=51, local_offset=-0.02),
        })

        # 5) Smooth + Sobel + Watershed
        variants.append({
            "name": "sobel_watershed",
            "params": dict(use_sobel_ws=True, peak_min_distance=10),
        })

    metrics_rows = []

    print(f"Processing frame {frame_index}...")
    for idx, variant in enumerate(variants, 1):
        name = variant["name"]
        params = variant["params"]
        print(f"  [{idx}/{len(variants)}] Testing variant: {name}")
        
        img, labels = segment_variant(frame, **params)

        fig = qc_overlay(img, labels, f"frame {frame_index} / {name}")
        out_png = output_dir / f"variant_{name}.png"
        fig.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close(fig)

        metrics = compute_metrics(img, labels)
        metrics["variant"] = name
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "variant_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(variants)} variants to: {output_dir}")
    print(f"Metrics saved to: {metrics_csv}")
    print(f"{'='*60}\n")
    
    # Print top results
    print("Top 5 variants by label count:")
    print(metrics_df.nlargest(5, "label_count")[["variant", "label_count", "coverage", "inside_outside_contrast"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation parameter sweep")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    main(Path(args.input), args.frame, Path(args.output_dir))
