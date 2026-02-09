"""
Test Cellpose segmentation on a single frame and save QC overlay.

Usage:
    uv run src/tests/cellpose_frame_test.py --input <tiff_path> --frame 1
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.restoration import rolling_ball
from skimage.morphology import remove_small_holes, binary_closing, disk
from skimage.segmentation import find_boundaries

try:
    from cellpose import models
except Exception as exc:
    raise SystemExit(f"Cellpose import failed: {exc}")


DEFAULT_INPUT = Path("data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff")
DEFAULT_OUTDIR = Path("results/cellpose")


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        frame = (frame - min_val) / (max_val - min_val)
    else:
        frame = np.zeros_like(frame)
    return frame


def get_model():
    # Handle cellpose 4.x API differences.
    if hasattr(models, "CellposeModel"):
        return models.CellposeModel(model_type="cyto3")
    return models.Cellpose(model_type="cyto3")


def preprocess(img, use_clahe=False, bg_subtract=None, rb_radius=50):
    if bg_subtract == "rolling_ball":
        background = rolling_ball(img, radius=rb_radius)
        img = img - background
        img = normalize_frame(img)

    if use_clahe:
        img = equalize_adapthist(img, clip_limit=0.02)
        img = normalize_frame(img)

    return img


def run_cellpose(img, diameter=30, cellprob_threshold=0.0, flow_threshold=0.4, min_size=0):
    model = get_model()
    # channels=[0,0] indicates grayscale
    if diameter is not None and diameter <= 0:
        diameter = 30

    result = model.eval(
        img,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )

    # Handle different return shapes across versions
    if isinstance(result, tuple):
        if len(result) == 4:
            masks, flows, styles, diams = result
        elif len(result) == 3:
            masks, flows, styles = result
        else:
            masks = result[0]
    else:
        masks = result

    return masks


def postprocess_masks(masks, fill_holes_area=0, closing_disk_size=0):
    processed = masks.copy()
    labels = np.unique(processed)
    for label_id in labels:
        if label_id == 0:
            continue
        region = processed == label_id
        if fill_holes_area > 0:
            region = remove_small_holes(region, area_threshold=fill_holes_area)
        if closing_disk_size > 0:
            region = binary_closing(region, disk(closing_disk_size))
        processed[processed == label_id] = 0
        processed[region] = label_id

    return processed


def qc_overlay(img, mask, title, brightness=1.25, gamma=0.9):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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


def main(
    input_path,
    frame_index,
    output_dir,
    diameter,
    use_clahe,
    bg_subtract,
    rb_radius,
    fill_holes_area,
    closing_disk_size,
    cellprob_threshold,
    flow_threshold,
    min_size,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]
    img = normalize_frame(frame)
    img = preprocess(img, use_clahe=use_clahe, bg_subtract=bg_subtract, rb_radius=rb_radius)

    masks = run_cellpose(
        img,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    if fill_holes_area > 0 or closing_disk_size > 0:
        masks = postprocess_masks(
            masks,
            fill_holes_area=fill_holes_area,
            closing_disk_size=closing_disk_size,
        )

    fig = qc_overlay(img, masks, f"cellpose frame {frame_index}")
    out_png = output_dir / (
        f"cellpose_frame_{frame_index}"
        f"_d{int(diameter)}"
        f"_cp{cellprob_threshold}"
        f"_ms{int(min_size)}"
        f"_fh{int(fill_holes_area)}"
        f"_cd{int(closing_disk_size)}"
        ".png"
    )
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellpose single-frame test")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--diameter", type=float, default=30)
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--bg-subtract", type=str, default="none", choices=["rolling_ball", "none"])
    parser.add_argument("--rb-radius", type=int, default=50)
    parser.add_argument("--fill-holes-area", type=int, default=0)
    parser.add_argument("--closing-disk", type=int, default=0)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--min-size", type=int, default=0)
    args = parser.parse_args()

    bg_subtract = args.bg_subtract
    if bg_subtract == "none":
        bg_subtract = None

    main(
        Path(args.input),
        args.frame,
        Path(args.output_dir),
        args.diameter,
        args.use_clahe,
        bg_subtract,
        args.rb_radius,
        args.fill_holes_area,
        args.closing_disk,
        args.cellprob_threshold,
        args.flow_threshold,
        args.min_size,
    )
