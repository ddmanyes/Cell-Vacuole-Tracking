"""
Test bubble detection on a single frame using two preprocessing paths:
1) Rolling ball + CLAHE
2) DoG + CLAHE

Usage:
    uv run src/tests/bubble_frame_preproc_test.py --input <tiff_path> --frame 0
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.morphology import binary_opening, disk, remove_small_objects
from skimage.restoration import rolling_ball
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from skimage.measure import regionprops
from scipy import ndimage

DEFAULT_INPUT = Path("data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff")
DEFAULT_OUTDIR = Path("results/bubble_preproc_frame")

CELLPOSE_MODEL_TYPE = "cyto3"
CELLPOSE_DIAMETER = 100
CELLPOSE_CELLPROB_THRESHOLD = 0.6
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_MIN_SIZE = 0
CELLPOSE_RB_RADIUS = 50
CELLPOSE_USE_CLAHE = True

BUBBLE_TH_THRESH = 0.28
BUBBLE_TH_MIN_AREA = 20
BUBBLE_TH_MAX_AREA = None
BUBBLE_TH_MIN_CIRCULARITY = 0.1

PREPROC_CLAHE_CLIP = 0.06
PREPROC_DOG_SIGMA1 = 1.0
PREPROC_DOG_SIGMA2 = 3.0


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


def preprocess_rb_clahe(frame: np.ndarray) -> np.ndarray:
    img = normalize_frame(frame)
    background = rolling_ball(img, radius=CELLPOSE_RB_RADIUS)
    img = img - background
    img = normalize_frame(img)
    img = equalize_adapthist(img, clip_limit=PREPROC_CLAHE_CLIP)
    return normalize_frame(img)


def preprocess_dog_clahe(frame: np.ndarray) -> np.ndarray:
    img = normalize_frame(frame)
    dog = gaussian(img, sigma=PREPROC_DOG_SIGMA1) - gaussian(img, sigma=PREPROC_DOG_SIGMA2)
    dog = normalize_frame(dog)
    dog = equalize_adapthist(dog, clip_limit=PREPROC_CLAHE_CLIP)
    return normalize_frame(dog)


def detect_bubbles_threshold(pre: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    binary = (pre < BUBBLE_TH_THRESH) & (cell_mask > 0)
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=BUBBLE_TH_MIN_AREA)

    labels = ndimage.label(binary)[0]
    if labels.max() == 0:
        return labels

    filtered = np.zeros_like(labels)
    idx = 1
    for prop in regionprops(labels):
        area = int(prop.area)
        if area < BUBBLE_TH_MIN_AREA:
            continue
        if BUBBLE_TH_MAX_AREA is not None and area > BUBBLE_TH_MAX_AREA:
            continue
        perimeter = float(prop.perimeter) if prop.perimeter > 0 else 1.0
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
        if circularity < BUBBLE_TH_MIN_CIRCULARITY:
            continue
        filtered[labels == prop.label] = idx
        idx += 1

    return filtered


def save_overlay(out_path: Path, img: np.ndarray, mask: np.ndarray, bubble_labels: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
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


def build_overlay(img: np.ndarray, mask: np.ndarray, bubble_labels: np.ndarray | None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    cell_outline = find_boundaries(mask, mode="outer")
    cell_outline = dilation(cell_outline, disk(1))
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = cell_outline.astype(np.float32)
    ax.imshow(overlay)

    if bubble_labels is not None:
        bubble_outline = find_boundaries(bubble_labels.astype(np.int32), mode="outer")
        bubble_outline = dilation(bubble_outline, disk(1))
        bubble_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        bubble_overlay[..., 0] = bubble_outline.astype(np.float32)
        bubble_overlay[..., 1] = bubble_outline.astype(np.float32)
        bubble_overlay[..., 2] = bubble_outline.astype(np.float32)
        bubble_overlay[..., 3] = bubble_outline.astype(np.float32)
        ax.imshow(bubble_overlay)

    ax.axis("off")
    fig.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    from PIL import Image
    return np.array(Image.open(buf))


def save_triptych(out_path: Path, img: np.ndarray, mask: np.ndarray, bubbles_rb: np.ndarray, bubbles_dog: np.ndarray) -> None:
    base = build_overlay(img, mask, None)
    rb = build_overlay(img, mask, bubbles_rb)
    dog = build_overlay(img, mask, bubbles_dog)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(base)
    axes[0].set_title("cells only")
    axes[1].imshow(rb)
    axes[1].set_title("rb + clahe")
    axes[2].imshow(dog)
    axes[2].set_title("dog + clahe")
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_bubble_stats(frame_mask: np.ndarray, bubble_labels: np.ndarray) -> list[dict]:
    results = []
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

    for cell_id in np.unique(frame_mask):
        if cell_id == 0:
            continue
        cell_region = frame_mask == cell_id
        cell_size = int(np.sum(cell_region))
        owned = [b_id for b_id, owner in bubble_owner.items() if owner == cell_id]
        bubble_count = int(len(owned))
        bubble_area = float(np.sum([bubble_area_map[b_id] for b_id in owned]))
        bubble_ratio = float(bubble_area / cell_size) if cell_size > 0 else 0.0
        results.append({
            "cell_id": int(cell_id),
            "cell_size_px": cell_size,
            "bubble_count": bubble_count,
            "bubble_area": bubble_area,
            "bubble_area_ratio": bubble_ratio,
        })

    return results


def write_stats_csv(out_path: Path, stats: list[dict]) -> None:
    if not stats:
        return
    fieldnames = list(stats[0].keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


def main(input_path: Path, frame_index: int, output_dir: Path, bubble_thresh: float | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]

    pre_cellpose = preprocess_cellpose_frame(frame)
    cell_mask = segment_cellpose(pre_cellpose)

    if bubble_thresh is not None:
        global BUBBLE_TH_THRESH
        BUBBLE_TH_THRESH = bubble_thresh

    pre_rb_clahe = preprocess_rb_clahe(frame)
    bubbles_rb_clahe = detect_bubbles_threshold(pre_rb_clahe, cell_mask)

    pre_dog_clahe = preprocess_dog_clahe(frame)
    bubbles_dog_clahe = detect_bubbles_threshold(pre_dog_clahe, cell_mask)

    stats_rb = compute_bubble_stats(cell_mask, bubbles_rb_clahe)
    stats_dog = compute_bubble_stats(cell_mask, bubbles_dog_clahe)

    write_stats_csv(output_dir / f"frame_{frame_index}_rb_clahe_stats.csv", stats_rb)
    write_stats_csv(output_dir / f"frame_{frame_index}_dog_clahe_stats.csv", stats_dog)

    save_overlay(
        output_dir / f"frame_{frame_index}_rb_clahe.png",
        normalize_frame(frame),
        cell_mask,
        bubbles_rb_clahe,
        f"frame {frame_index} / rb+clahe",
    )
    save_overlay(
        output_dir / f"frame_{frame_index}_dog_clahe.png",
        normalize_frame(frame),
        cell_mask,
        bubbles_dog_clahe,
        f"frame {frame_index} / dog+clahe",
    )

    save_triptych(
        output_dir / f"frame_{frame_index}_compare.png",
        normalize_frame(frame),
        cell_mask,
        bubbles_rb_clahe,
        bubbles_dog_clahe,
    )

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble detection preprocessing test")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bubble-thresh", type=float, default=None)
    args = parser.parse_args()

    main(Path(args.input), args.frame, Path(args.output_dir), args.bubble_thresh)
