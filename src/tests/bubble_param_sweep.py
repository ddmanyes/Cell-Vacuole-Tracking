"""
Bubble detection parameter sweep for visual QC and metrics.

This script automates the testing of multiple bubble detection parameter combinations
on a single frame, generating both visual overlays and quantitative metrics for comparison.

Features:
- Automatically loads baseline parameters from config/pipeline_params.yaml
- Tests systematic sweeps across threshold, min_area, and CLAHE clip parameters
- Includes predefined "sensitive_small" and "balanced" preset combinations
- Generates QC overlays for visual inspection
- Exports CSV metrics for quantitative comparison

Parameter Sweeps:
1. Baseline: Default parameters from config (serves as reference)
2. Threshold sweep: Tests detection sensitivity (lower = stricter, darker bubbles only)
3. Min area sweep: Tests size filtering (smaller = more sensitive to small bubbles)
4. CLAHE clip sweep: Tests contrast enhancement (higher = stronger enhancement)
5. Recommended presets: Optimized combinations for common use cases

Sweep ranges are configurable in config/pipeline_params.yaml under 'bubble_sweep':
    bubble_sweep:
      thresh: [0.25, 0.30, 0.35, 0.40]       # Detection threshold values
      min_area: [5, 10, 15, 20]               # Minimum bubble area (pixels)
      clahe_clip: [0.04, 0.06, 0.08, 0.10, 0.15]  # CLAHE contrast limits

Output Files:
- results/bubble_param_sweep/<variant_name>.png: 
    QC overlay showing cells (blue outline) and detected bubbles (white outline)
- results/bubble_param_sweep/bubble_param_metrics.csv:
    Quantitative metrics for each parameter combination including:
      * variant: Parameter set name
      * total_bubbles: Total number of detected bubbles
      * total_bubble_area: Sum of all bubble areas (pixels)
      * cells_with_bubbles: Number of cells containing ≥1 bubble
      * total_cells: Total number of segmented cells
      * avg_bubbles_per_cell: Mean bubbles per cell
      * avg_bubble_area: Mean bubble size (pixels)
      * thresh, min_area, clahe_clip, rb_radius: Parameter values used

Usage:
    # Basic usage (frame 0 of default input)
    uv run src/tests/bubble_param_sweep.py --input <tiff_path> --frame 0
    
    # Custom output directory
    uv run src/tests/bubble_param_sweep.py --input <tiff_path> --frame 0 --output-dir results/custom_sweep
    
    # Test different frames to verify consistency
    uv run src/tests/bubble_param_sweep.py --input <tiff_path> --frame 5
    uv run src/tests/bubble_param_sweep.py --input <tiff_path> --frame 10

Workflow:
1. Run the sweep on a representative frame (e.g., frame 0)
2. Review bubble_param_metrics.csv to identify promising parameter sets
3. Visually inspect corresponding PNG overlays to validate detection quality
4. Test selected parameters on additional frames to verify consistency
5. Update config/pipeline_params.yaml with optimal parameters
6. Run full pipeline with optimized settings

Tips:
- Start with the baseline to understand current detection performance
- Compare metrics across sweeps to identify parameter sensitivity
- Balance avg_bubbles_per_cell against biological expectations
- Check avg_bubble_area to ensure detected sizes are plausible
- The script prints top 5 variants by bubble count for quick reference
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_opening, disk, remove_small_objects
from skimage.restoration import rolling_ball
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from skimage.measure import regionprops
from scipy import ndimage

import yaml
import sys

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

DEFAULT_OUTDIR = Path("results/bubble_param_sweep")

# Cellpose defaults (override with config if present)
cp_conf = CONFIG.get("cellpose", {})
CELLPOSE_MODEL_TYPE = cp_conf.get("model_type", "cyto3")
CELLPOSE_DIAMETER = cp_conf.get("diameter", 100)
CELLPOSE_CELLPROB_THRESHOLD = cp_conf.get("cellprob_threshold", 0.6)
CELLPOSE_FLOW_THRESHOLD = cp_conf.get("flow_threshold", 0.4)
CELLPOSE_MIN_SIZE = cp_conf.get("min_size", 0)
CELLPOSE_RB_RADIUS = cp_conf.get("rb_radius", 50)
CELLPOSE_USE_CLAHE = cp_conf.get("use_clahe", True)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        frame = (frame - min_val) / (max_val - min_val)
    else:
        frame = np.zeros_like(frame)
    return frame


def preprocess_cellpose_frame(frame: np.ndarray) -> np.ndarray:
    img = normalize_frame(frame)
    background = rolling_ball(img, radius=CELLPOSE_RB_RADIUS)
    img = img - background
    img = normalize_frame(img)

    if CELLPOSE_USE_CLAHE:
        img = equalize_adapthist(img, clip_limit=0.02)
        img = normalize_frame(img)

    return img


def segment_cellpose(img: np.ndarray) -> np.ndarray:
    model = models.CellposeModel(model_type=CELLPOSE_MODEL_TYPE)
    result = model.eval(
        img,
        channels=[0, 0],
        diameter=CELLPOSE_DIAMETER,
        cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
        flow_threshold=CELLPOSE_FLOW_THRESHOLD,
        min_size=CELLPOSE_MIN_SIZE,
    )

    masks = result[0] if isinstance(result, tuple) else result
    return masks.astype(np.int32)


def preprocess_rb_clahe(frame: np.ndarray, clahe_clip: float, rb_radius: int) -> np.ndarray:
    img = normalize_frame(frame)
    background = rolling_ball(img, radius=rb_radius)
    img = img - background
    img = normalize_frame(img)
    img = equalize_adapthist(img, clip_limit=clahe_clip)
    return normalize_frame(img)


def detect_bubbles_threshold(
    pre: np.ndarray,
    cell_mask: np.ndarray,
    thresh: float,
    min_area: int,
    max_area: int | None,
    min_circularity: float,
) -> np.ndarray:
    binary = (pre < thresh) & (cell_mask > 0)
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=min_area)

    labels = ndimage.label(binary)[0]
    if labels.max() == 0:
        return labels

    filtered = np.zeros_like(labels)
    idx = 1
    for prop in regionprops(labels):
        area = int(prop.area)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        perimeter = float(prop.perimeter) if prop.perimeter > 0 else 1.0
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
        if circularity < min_circularity:
            continue
        filtered[labels == prop.label] = idx
        idx += 1

    return filtered


def compute_bubble_stats(frame_mask: np.ndarray, bubble_labels: np.ndarray) -> dict:
    """Compute bubble detection statistics."""
    bubble_ids = np.unique(bubble_labels)
    bubble_ids = bubble_ids[bubble_ids != 0]
    
    bubble_owner = {}
    bubble_area_map = {}
    for bubble_id in bubble_ids:
        bubble_mask = bubble_labels == bubble_id
        mask_vals = frame_mask[bubble_mask]
        unique_cells = np.unique(mask_vals)
        if len(unique_cells) == 1 and unique_cells[0] != 0:
            bubble_owner[bubble_id] = int(unique_cells[0])
            bubble_area_map[bubble_id] = int(np.sum(bubble_mask))
        else:
            bubble_owner[bubble_id] = 0
            bubble_area_map[bubble_id] = int(np.sum(bubble_mask))

    total_bubbles = len(bubble_ids)
    total_bubble_area = sum(bubble_area_map.values())
    cells_with_bubbles = len(set(bubble_owner.values()) - {0})
    
    total_cells = len(np.unique(frame_mask)) - 1  # exclude background
    
    return {
        "total_bubbles": total_bubbles,
        "total_bubble_area": total_bubble_area,
        "cells_with_bubbles": cells_with_bubbles,
        "total_cells": total_cells,
        "avg_bubbles_per_cell": total_bubbles / total_cells if total_cells > 0 else 0.0,
        "avg_bubble_area": total_bubble_area / total_bubbles if total_bubbles > 0 else 0.0,
    }


def save_overlay(
    out_path: Path,
    img: np.ndarray,
    mask: np.ndarray,
    bubble_labels: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")

    cell_outline = find_boundaries(mask, mode="outer")
    cell_outline = dilation(cell_outline, disk(1))
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = cell_outline.astype(np.float32)
    ax.imshow(overlay)

    bubble_outline = find_boundaries(bubble_labels.astype(np.int32), mode="outer")
    bubble_outline = dilation(bubble_outline, disk(1))
    bubble_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    bubble_overlay[..., 0] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 1] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 2] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 3] = bubble_outline.astype(np.float32)
    ax.imshow(bubble_overlay)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(input_path: Path, frame_index: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]

    # Segment cells once
    print("Segmenting cells with Cellpose...")
    pre_cellpose = preprocess_cellpose_frame(frame)
    cell_mask = segment_cellpose(pre_cellpose)

    # Define parameter sweep from config
    sweep_conf = CONFIG.get("bubble_sweep", {})
    thresh_sweep = sweep_conf.get("thresh", [0.25, 0.30, 0.35, 0.40])
    min_area_sweep = sweep_conf.get("min_area", [5, 10, 15, 20])
    clahe_clip_sweep = sweep_conf.get("clahe_clip", [0.04, 0.06, 0.08, 0.10, 0.15])

    variants = []

    # Baseline from config
    bubble_conf = CONFIG.get("bubble", {}).get("rb_clahe", {})
    baseline_thresh = bubble_conf.get("thresh", 0.28)
    baseline_min_area = bubble_conf.get("min_area", 20)
    baseline_clahe = bubble_conf.get("clahe_clip", 0.06)
    baseline_rb = bubble_conf.get("rb_radius", 50)
    
    # 1) Baseline
    variants.append({
        "name": "baseline",
        "thresh": baseline_thresh,
        "min_area": baseline_min_area,
        "clahe_clip": baseline_clahe,
        "rb_radius": baseline_rb,
    })

    # 2) Threshold sweep (keeping other params constant)
    for thresh in thresh_sweep:
        variants.append({
            "name": f"thresh_{thresh:.2f}",
            "thresh": thresh,
            "min_area": baseline_min_area,
            "clahe_clip": baseline_clahe,
            "rb_radius": baseline_rb,
        })

    # 3) Min area sweep
    for min_area in min_area_sweep:
        variants.append({
            "name": f"min_area_{min_area}",
            "thresh": baseline_thresh,
            "min_area": min_area,
            "clahe_clip": baseline_clahe,
            "rb_radius": baseline_rb,
        })

    # 4) CLAHE clip sweep
    for clahe_clip in clahe_clip_sweep:
        variants.append({
            "name": f"clahe_{clahe_clip:.2f}",
            "thresh": baseline_thresh,
            "min_area": baseline_min_area,
            "clahe_clip": clahe_clip,
            "rb_radius": baseline_rb,
        })

    # 5) Recommended combinations
    variants.append({
        "name": "sensitive_small",
        "thresh": 0.35,
        "min_area": 5,
        "clahe_clip": 0.10,
        "rb_radius": 50,
    })
    
    variants.append({
        "name": "balanced",
        "thresh": 0.30,
        "min_area": 10,
        "clahe_clip": 0.08,
        "rb_radius": 50,
    })

    metrics_rows = []
    
    print(f"Testing {len(variants)} parameter combinations...")
    for variant in variants:
        name = variant["name"]
        thresh = variant["thresh"]
        min_area = variant["min_area"]
        clahe_clip = variant["clahe_clip"]
        rb_radius = variant["rb_radius"]

        # Preprocess and detect bubbles
        pre = preprocess_rb_clahe(frame, clahe_clip, rb_radius)
        bubbles = detect_bubbles_threshold(
            pre,
            cell_mask,
            thresh=thresh,
            min_area=min_area,
            max_area=None,
            min_circularity=0.1,
        )

        # Save overlay
        out_png = output_dir / f"{name}.png"
        save_overlay(
            out_png,
            normalize_frame(frame),
            cell_mask,
            bubbles,
            f"frame {frame_index} / {name}",
        )

        # Compute metrics
        stats = compute_bubble_stats(cell_mask, bubbles)
        stats["variant"] = name
        stats["thresh"] = thresh
        stats["min_area"] = min_area
        stats["clahe_clip"] = clahe_clip
        stats["rb_radius"] = rb_radius
        metrics_rows.append(stats)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "bubble_param_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(variants)} variants to: {output_dir}")
    print(f"Metrics saved to: {metrics_csv}")
    print(f"{'='*60}\n")
    
    # Print top results
    print("Top 5 variants by total bubbles detected:")
    print(metrics_df.nlargest(5, "total_bubbles")[["variant", "total_bubbles", "avg_bubbles_per_cell", "thresh", "min_area", "clahe_clip"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble detection parameter sweep")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    main(Path(args.input), args.frame, Path(args.output_dir))
