"""
Test bubble detection on a single Cellpose-segmented cell.

Usage:
    uv run src/tests/bubble_single_cell_test.py --input <tiff_path> --frame 0
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

import cv2
from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, black_tophat
from skimage.restoration import rolling_ball
from skimage.segmentation import find_boundaries
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

DEFAULT_OUTDIR = Path("results/bubble_single_cell")

# Cellpose defaults (override with config if present)
cp_conf = CONFIG.get("cellpose", {})
CELLPOSE_MODEL_TYPE = cp_conf.get("model_type", "cyto3")
CELLPOSE_DIAMETER = cp_conf.get("diameter", 100)
CELLPOSE_CELLPROB_THRESHOLD = cp_conf.get("cellprob_threshold", 0.6)
CELLPOSE_FLOW_THRESHOLD = cp_conf.get("flow_threshold", 0.4)
CELLPOSE_MIN_SIZE = cp_conf.get("min_size", 0)
CELLPOSE_RB_RADIUS = cp_conf.get("rb_radius", 50)
CELLPOSE_USE_CLAHE = cp_conf.get("use_clahe", True)
CELLPOSE_FILL_HOLES_AREA = cp_conf.get("fill_holes_area", 0)
CELLPOSE_CLOSING_DISK = cp_conf.get("closing_disk", 4)

# Bubble defaults (override with config if present)
bubble_conf = CONFIG.get("bubble", {}).get("rb_clahe", {})
BUBBLE_TH_THRESH_SWEEP = [bubble_conf.get("thresh", 0.28)]  # Default to config value
if not bubble_conf: # Fallback sweep range if config missing
    BUBBLE_TH_THRESH_SWEEP = [0.15, 0.2, 0.25]

# If config exists, center sweep around the config value
conf_thresh = bubble_conf.get("thresh", 0.28)
BUBBLE_TH_THRESH_SWEEP = [max(0.01, conf_thresh - 0.05), conf_thresh, min(1.0, conf_thresh + 0.05)]

BUBBLE_TH_MIN_AREA = bubble_conf.get("min_area", 20)
BUBBLE_TH_MAX_AREA = bubble_conf.get("max_area", 400)
if BUBBLE_TH_MAX_AREA is None: BUBBLE_TH_MAX_AREA = 10000

BUBBLE_TH_MIN_CIRCULARITY = bubble_conf.get("min_circularity", 0.1)
PREPROC_CLAHE_CLIP = bubble_conf.get("clahe_clip", 0.06)

PREPROC_RB_RADIUS = bubble_conf.get("rb_radius", 50)

# Other constants remain defaults for now
BUBBLE_CELLPOSE_MODEL_TYPE = "cyto3"
BUBBLE_CELLPOSE_DIAMETER = 10
BUBBLE_CELLPOSE_CELLPROB_THRESHOLD = 0.0
BUBBLE_CELLPOSE_FLOW_THRESHOLD = 0.4
BUBBLE_CELLPOSE_MIN_SIZE = 0
BUBBLE_CELLPOSE_SMOOTH_SIGMA = 1.0
BUBBLE_CELLPOSE_INVERT = True
BUBBLE_CELLPOSE_DIAMETER_SWEEP = [8]
BUBBLE_CELLPOSE_FLOW_THRESHOLD_SWEEP = [0.1, 0.2, 0.4, 0.6]
BUBBLE_CELLPOSE_RB_RADIUS_SWEEP = [5, 50]
BUBBLE_CELLPOSE_CLAHE_CLIP_SWEEP = [0.02, 0.06]
BUBBLE_CELLPOSE_SMOOTH_SIGMA_SWEEP = [1.0, 1.5, 2.0]

BUBBLE_TH_SMOOTH_SIGMA = 1.0
BUBBLE_TH_CIRC_SWEEP = [0.5, 0.6, 0.7]

BUBBLE_HOUGH_BLUR = 3
BUBBLE_HOUGH_MIN_RADIUS = 3
BUBBLE_HOUGH_MAX_RADIUS = 12
BUBBLE_HOUGH_PARAM1_SWEEP = [50, 100, 150]
BUBBLE_HOUGH_PARAM2_SWEEP = [8, 12, 16]

PREPROC_BLACKHAT_RADIUS = 5
PREPROC_DOG_SIGMA1 = 1.0
PREPROC_DOG_SIGMA2 = 3.0
PREPROC_DOG_CLAHE_THRESHOLD_SWEEP = [0.15, 0.2, 0.25]
PREPROC_DOG_CLAHE_CIRC_SWEEP = [0.5, 0.6, 0.7]


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


def postprocess_masks(mask: np.ndarray) -> np.ndarray:
    processed = mask.copy()
    for label_id in np.unique(processed):
        if label_id == 0:
            continue
        region = processed == label_id
        if CELLPOSE_FILL_HOLES_AREA > 0:
            region = ndimage.binary_fill_holes(region)
        if CELLPOSE_CLOSING_DISK > 0:
            region = binary_closing(region, disk(CELLPOSE_CLOSING_DISK))
        processed[processed == label_id] = 0
        processed[region] = label_id
    return processed


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
    return postprocess_masks(masks.astype(np.int32))


def preprocess_bubble_cellpose_frame(
    frame: np.ndarray,
    rb_radius: int,
    use_clahe: bool,
    clahe_clip: float,
    smooth_sigma: float,
    invert: bool,
) -> np.ndarray:
    img = normalize_frame(frame)
    if rb_radius and rb_radius > 0:
        background = rolling_ball(img, radius=rb_radius)
        img = img - background
        img = normalize_frame(img)
    if use_clahe:
        img = equalize_adapthist(img, clip_limit=clahe_clip)
        img = normalize_frame(img)
    if smooth_sigma and smooth_sigma > 0:
        img = gaussian(img, sigma=smooth_sigma)
    if invert:
        img = 1.0 - img
        img = normalize_frame(img)
    return img


def preprocess_blackhat(img: np.ndarray, radius: int) -> np.ndarray:
    return black_tophat(img, disk(radius))


def preprocess_dog(img: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    blurred1 = gaussian(img, sigma=sigma1)
    blurred2 = gaussian(img, sigma=sigma2)
    return blurred1 - blurred2


def preprocess_dog_clahe(img: np.ndarray) -> np.ndarray:
    base = normalize_frame(img)
    dog = preprocess_dog(base, PREPROC_DOG_SIGMA1, PREPROC_DOG_SIGMA2)
    dog = normalize_frame(dog)
    dog = equalize_adapthist(dog, clip_limit=PREPROC_CLAHE_CLIP)
    return normalize_frame(dog)


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    base = normalize_frame(img)
    clahe = equalize_adapthist(base, clip_limit=PREPROC_CLAHE_CLIP)
    return normalize_frame(clahe)


def save_preprocess_grid(out_path: Path, cell_img: np.ndarray) -> None:
    base = normalize_frame(cell_img)

    rb = normalize_frame(base - rolling_ball(base, radius=PREPROC_RB_RADIUS))
    clahe = normalize_frame(equalize_adapthist(base, clip_limit=PREPROC_CLAHE_CLIP))
    rb_clahe = normalize_frame(equalize_adapthist(rb, clip_limit=PREPROC_CLAHE_CLIP))

    blackhat = normalize_frame(preprocess_blackhat(base, PREPROC_BLACKHAT_RADIUS))
    blackhat_rb = normalize_frame(preprocess_blackhat(rb, PREPROC_BLACKHAT_RADIUS))

    dog = normalize_frame(preprocess_dog(base, PREPROC_DOG_SIGMA1, PREPROC_DOG_SIGMA2))
    dog_rb = normalize_frame(preprocess_dog(rb, PREPROC_DOG_SIGMA1, PREPROC_DOG_SIGMA2))

    variants = [
        ("base", base),
        ("rb", rb),
        ("clahe", clahe),
        ("rb+clahe", rb_clahe),
        ("blackhat", blackhat),
        ("blackhat+rb", blackhat_rb),
        ("dog", dog),
        ("dog+rb", dog_rb),
    ]

    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for idx, (title, img) in enumerate(variants):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def detect_bubbles_cellpose(
    cell_img: np.ndarray,
    cell_mask: np.ndarray,
    diameter: int,
    flow_threshold: float,
    rb_radius: int,
    use_clahe: bool,
    clahe_clip: float,
    smooth_sigma: float,
) -> np.ndarray:
    if models is None:
        raise RuntimeError("Cellpose is not available. Install cellpose or change BUBBLE_METHOD.")

    img = preprocess_bubble_cellpose_frame(
        cell_img,
        rb_radius=rb_radius,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        smooth_sigma=smooth_sigma,
        invert=BUBBLE_CELLPOSE_INVERT,
    )
    img = img * cell_mask

    model = models.CellposeModel(model_type=BUBBLE_CELLPOSE_MODEL_TYPE)
    result = model.eval(
        img,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=BUBBLE_CELLPOSE_CELLPROB_THRESHOLD,
        flow_threshold=flow_threshold,
        min_size=BUBBLE_CELLPOSE_MIN_SIZE,
    )

    masks = result[0] if isinstance(result, tuple) else result
    masks = masks.astype(np.int32)
    masks *= cell_mask.astype(np.int32)
    return masks


def detect_bubbles_threshold(
    cell_img: np.ndarray,
    cell_mask: np.ndarray,
    thresh: float,
    min_area: int,
    max_area: int,
    min_circularity: float,
    smooth_sigma: float,
) -> np.ndarray:
    img = normalize_frame(cell_img)
    if smooth_sigma and smooth_sigma > 0:
        img = gaussian(img, sigma=smooth_sigma)

    binary = (img < thresh) & cell_mask
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=min_area)

    labels = ndimage.label(binary)[0]
    if labels.max() == 0:
        return labels

    filtered = np.zeros_like(labels)
    idx = 1
    for prop in regionprops(labels):
        area = int(prop.area)
        if area < min_area or area > max_area:
            continue
        perimeter = float(prop.perimeter) if prop.perimeter > 0 else 1.0
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
        if circularity < min_circularity:
            continue
        filtered[labels == prop.label] = idx
        idx += 1

    return filtered


def detect_bubbles_threshold_from_preprocessed(
    pre: np.ndarray,
    cell_mask: np.ndarray,
    thresh: float,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> np.ndarray:
    binary = (pre < thresh) & cell_mask
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=min_area)

    labels = ndimage.label(binary)[0]
    if labels.max() == 0:
        return labels

    filtered = np.zeros_like(labels)
    idx = 1
    for prop in regionprops(labels):
        area = int(prop.area)
        if area < min_area or area > max_area:
            continue
        perimeter = float(prop.perimeter) if prop.perimeter > 0 else 1.0
        circularity = float(4.0 * np.pi * area / (perimeter ** 2))
        if circularity < min_circularity:
            continue
        filtered[labels == prop.label] = idx
        idx += 1

    return filtered


def detect_bubbles_hough(
    cell_img: np.ndarray,
    cell_mask: np.ndarray,
    param1: float,
    param2: float,
    min_radius: int,
    max_radius: int,
) -> np.ndarray:
    img = normalize_frame(cell_img)
    img = gaussian(img, sigma=1.0)
    img = (img * 255).astype(np.uint8)
    img[~cell_mask] = 0

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=8,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return np.array([]).reshape(0, 3)

    return circles[0]


def save_overlay(out_path: Path, cell_img: np.ndarray, cell_mask: np.ndarray, bubble_masks: np.ndarray, title: str) -> None:
    bubble_outline = find_boundaries(bubble_masks.astype(np.int32), mode="outer")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cell_img, cmap="gray")

    cell_outline = find_boundaries(cell_mask, mode="outer")
    overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = cell_outline.astype(np.float32)
    ax.imshow(overlay)

    bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    bubble_overlay[..., 0] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 1] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 2] = bubble_outline.astype(np.float32)
    bubble_overlay[..., 3] = bubble_outline.astype(np.float32)
    ax.imshow(bubble_overlay)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_hough_outline(shape: tuple[int, int], circles: np.ndarray) -> np.ndarray:
    outline = np.zeros(shape, dtype=np.uint8)
    if circles.size == 0:
        return outline.astype(bool)
    for x, y, r in circles:
        cv2.circle(outline, (int(round(x)), int(round(y))), int(round(r)), 1, thickness=1)
    return outline.astype(bool)


def save_threshold_sweep_grid(out_path: Path, cell_img: np.ndarray, cell_mask: np.ndarray) -> None:
    rows = len(BUBBLE_TH_THRESH_SWEEP)
    cols = len(BUBBLE_TH_CIRC_SWEEP)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])

    for r, thresh in enumerate(BUBBLE_TH_THRESH_SWEEP):
        for c, circ in enumerate(BUBBLE_TH_CIRC_SWEEP):
            masks = detect_bubbles_threshold(
                cell_img,
                cell_mask,
                thresh=thresh,
                min_area=BUBBLE_TH_MIN_AREA,
                max_area=BUBBLE_TH_MAX_AREA,
                min_circularity=circ,
                smooth_sigma=BUBBLE_TH_SMOOTH_SIGMA,
            )
            outline = find_boundaries(masks.astype(np.int32), mode="outer")
            ax = axes[r, c]
            ax.imshow(cell_img, cmap="gray")

            cell_outline = find_boundaries(cell_mask, mode="outer")
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = cell_outline.astype(np.float32)
            ax.imshow(overlay)

            bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            bubble_overlay[..., 0] = outline.astype(np.float32)
            bubble_overlay[..., 1] = outline.astype(np.float32)
            bubble_overlay[..., 2] = outline.astype(np.float32)
            bubble_overlay[..., 3] = outline.astype(np.float32)
            ax.imshow(bubble_overlay)

            ax.set_title(f"thr {thresh:.2f}\ncirc {circ:.2f}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_dog_clahe_threshold_sweep_grid(out_path: Path, cell_img: np.ndarray, cell_mask: np.ndarray) -> None:
    pre = preprocess_dog_clahe(cell_img)
    rows = len(PREPROC_DOG_CLAHE_THRESHOLD_SWEEP)
    cols = len(PREPROC_DOG_CLAHE_CIRC_SWEEP)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])

    for r, thresh in enumerate(PREPROC_DOG_CLAHE_THRESHOLD_SWEEP):
        for c, circ in enumerate(PREPROC_DOG_CLAHE_CIRC_SWEEP):
            masks = detect_bubbles_threshold_from_preprocessed(
                pre,
                cell_mask,
                thresh=thresh,
                min_area=BUBBLE_TH_MIN_AREA,
                max_area=BUBBLE_TH_MAX_AREA,
                min_circularity=circ,
            )
            outline = find_boundaries(masks.astype(np.int32), mode="outer")
            ax = axes[r, c]
            ax.imshow(cell_img, cmap="gray")

            cell_outline = find_boundaries(cell_mask, mode="outer")
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = cell_outline.astype(np.float32)
            ax.imshow(overlay)

            bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            bubble_overlay[..., 0] = outline.astype(np.float32)
            bubble_overlay[..., 1] = outline.astype(np.float32)
            bubble_overlay[..., 2] = outline.astype(np.float32)
            bubble_overlay[..., 3] = outline.astype(np.float32)
            ax.imshow(bubble_overlay)

            ax.set_title(f"dog+clahe\nthr {thresh:.2f}\ncirc {circ:.2f}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_clahe_threshold_sweep_grid(out_path: Path, cell_img: np.ndarray, cell_mask: np.ndarray) -> None:
    pre = preprocess_clahe(cell_img)
    rows = len(PREPROC_DOG_CLAHE_THRESHOLD_SWEEP)
    cols = len(PREPROC_DOG_CLAHE_CIRC_SWEEP)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])

    for r, thresh in enumerate(PREPROC_DOG_CLAHE_THRESHOLD_SWEEP):
        for c, circ in enumerate(PREPROC_DOG_CLAHE_CIRC_SWEEP):
            masks = detect_bubbles_threshold_from_preprocessed(
                pre,
                cell_mask,
                thresh=thresh,
                min_area=BUBBLE_TH_MIN_AREA,
                max_area=BUBBLE_TH_MAX_AREA,
                min_circularity=circ,
            )
            outline = find_boundaries(masks.astype(np.int32), mode="outer")
            ax = axes[r, c]
            ax.imshow(cell_img, cmap="gray")

            cell_outline = find_boundaries(cell_mask, mode="outer")
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = cell_outline.astype(np.float32)
            ax.imshow(overlay)

            bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            bubble_overlay[..., 0] = outline.astype(np.float32)
            bubble_overlay[..., 1] = outline.astype(np.float32)
            bubble_overlay[..., 2] = outline.astype(np.float32)
            bubble_overlay[..., 3] = outline.astype(np.float32)
            ax.imshow(bubble_overlay)

            ax.set_title(f"clahe\nthr {thresh:.2f}\ncirc {circ:.2f}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hough_sweep_grid(out_path: Path, cell_img: np.ndarray, cell_mask: np.ndarray) -> None:
    rows = len(BUBBLE_HOUGH_PARAM1_SWEEP)
    cols = len(BUBBLE_HOUGH_PARAM2_SWEEP)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])

    for r, param1 in enumerate(BUBBLE_HOUGH_PARAM1_SWEEP):
        for c, param2 in enumerate(BUBBLE_HOUGH_PARAM2_SWEEP):
            circles = detect_bubbles_hough(
                cell_img,
                cell_mask,
                param1=param1,
                param2=param2,
                min_radius=BUBBLE_HOUGH_MIN_RADIUS,
                max_radius=BUBBLE_HOUGH_MAX_RADIUS,
            )
            outline = draw_hough_outline(cell_mask.shape, circles)
            ax = axes[r, c]
            ax.imshow(cell_img, cmap="gray")

            cell_outline = find_boundaries(cell_mask, mode="outer")
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = cell_outline.astype(np.float32)
            ax.imshow(overlay)

            bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            bubble_overlay[..., 0] = outline.astype(np.float32)
            bubble_overlay[..., 1] = outline.astype(np.float32)
            bubble_overlay[..., 2] = outline.astype(np.float32)
            bubble_overlay[..., 3] = outline.astype(np.float32)
            ax.imshow(bubble_overlay)

            ax.set_title(f"p1 {param1}\np2 {param2}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cellpose_sweep_grid(
    out_path: Path,
    cell_img: np.ndarray,
    cell_mask: np.ndarray,
    smooth_sigma: float,
    clahe_clip: float,
) -> None:
    rows = len(BUBBLE_CELLPOSE_DIAMETER_SWEEP)
    cols = len(BUBBLE_CELLPOSE_FLOW_THRESHOLD_SWEEP) * len(BUBBLE_CELLPOSE_RB_RADIUS_SWEEP)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])

    col_labels = []
    for flow in BUBBLE_CELLPOSE_FLOW_THRESHOLD_SWEEP:
        for rb_radius in BUBBLE_CELLPOSE_RB_RADIUS_SWEEP:
            col_labels.append((flow, rb_radius))

    for r, diameter in enumerate(BUBBLE_CELLPOSE_DIAMETER_SWEEP):
        for c, (flow, rb_radius) in enumerate(col_labels):
            masks = detect_bubbles_cellpose(
                cell_img,
                cell_mask,
                diameter=diameter,
                flow_threshold=flow,
                rb_radius=rb_radius,
                use_clahe=True,
                clahe_clip=clahe_clip,
                smooth_sigma=smooth_sigma,
            )
            outline = find_boundaries(masks.astype(np.int32), mode="outer")
            ax = axes[r, c]
            ax.imshow(cell_img, cmap="gray")
            cell_outline = find_boundaries(cell_mask, mode="outer")
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = cell_outline.astype(np.float32)
            ax.imshow(overlay)

            bubble_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            bubble_overlay[..., 0] = outline.astype(np.float32)
            bubble_overlay[..., 1] = outline.astype(np.float32)
            bubble_overlay[..., 2] = outline.astype(np.float32)
            bubble_overlay[..., 3] = outline.astype(np.float32)
            ax.imshow(bubble_overlay)

            ax.set_title(f"d{diameter} f{flow:.1f}\nrb{rb_radius}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    input_path: Path,
    frame_index: int,
    output_dir: Path,
    bubble_smooth_sigma: float,
    cellpose_sweep: bool,
    threshold_sweep: bool,
    hough_sweep: bool,
    preprocess_sweep: bool,
    dog_clahe_sweep: bool,
    clahe_sweep: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]

    pre = preprocess_cellpose_frame(frame)
    mask = segment_cellpose(pre)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        raise SystemExit("No cells found in frame.")

    areas = [(label_id, int(np.sum(mask == label_id))) for label_id in labels]
    label_id, _ = max(areas, key=lambda x: x[1])

    cell_region = mask == label_id
    ys, xs = np.where(cell_region)
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1

    pad = 5
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(frame.shape[0], y_max + pad)
    x_max = min(frame.shape[1], x_max + pad)

    cell_img = frame[y_min:y_max, x_min:x_max]
    cell_mask = cell_region[y_min:y_max, x_min:x_max]

    smooth_tag = f"s{bubble_smooth_sigma:g}" if bubble_smooth_sigma else "s0"

    ran_sweep = False

    if preprocess_sweep:
        grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_preprocess_grid.png"
        save_preprocess_grid(grid_path, cell_img)
        print(f"Saved: {grid_path}")
        ran_sweep = True

    if dog_clahe_sweep:
        grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_dog_clahe_sweep.png"
        save_dog_clahe_threshold_sweep_grid(grid_path, cell_img, cell_mask)
        print(f"Saved: {grid_path}")
        ran_sweep = True

    if clahe_sweep:
        grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_clahe_sweep.png"
        save_clahe_threshold_sweep_grid(grid_path, cell_img, cell_mask)
        print(f"Saved: {grid_path}")
        ran_sweep = True

    if threshold_sweep:
        grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_threshold_sweep.png"
        save_threshold_sweep_grid(grid_path, cell_img, cell_mask)
        print(f"Saved: {grid_path}")
        ran_sweep = True

    if hough_sweep:
        grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_hough_sweep.png"
        save_hough_sweep_grid(grid_path, cell_img, cell_mask)
        print(f"Saved: {grid_path}")
        ran_sweep = True

    if cellpose_sweep:
        for smooth_sigma in BUBBLE_CELLPOSE_SMOOTH_SIGMA_SWEEP:
            smooth_tag = f"s{smooth_sigma:g}" if smooth_sigma else "s0"
            for clahe_clip in BUBBLE_CELLPOSE_CLAHE_CLIP_SWEEP:
                clahe_tag = f"clahe{clahe_clip:.2f}"
                grid_path = output_dir / f"cell_{label_id}_frame_{frame_index}_{smooth_tag}_{clahe_tag}_sweep.png"
                save_cellpose_sweep_grid(
                    grid_path,
                    cell_img,
                    cell_mask,
                    smooth_sigma,
                    clahe_clip=clahe_clip,
                )
                print(f"Saved: {grid_path}")
        ran_sweep = True

    if ran_sweep:
        return

    masks = detect_bubbles_cellpose(
        cell_img,
        cell_mask,
        diameter=BUBBLE_CELLPOSE_DIAMETER,
        flow_threshold=BUBBLE_CELLPOSE_FLOW_THRESHOLD,
        rb_radius=CELLPOSE_RB_RADIUS,
        use_clahe=CELLPOSE_USE_CLAHE,
        clahe_clip=0.02,
        smooth_sigma=bubble_smooth_sigma,
    )
    out_path = output_dir / f"cell_{label_id}_frame_{frame_index}_{smooth_tag}.png"
    save_overlay(out_path, cell_img, cell_mask, masks, f"frame {frame_index} / cell {label_id}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single cell bubble test")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bubble-smooth-sigma", type=float, default=BUBBLE_CELLPOSE_SMOOTH_SIGMA)
    parser.add_argument("--sweep", action="store_true", help="Alias for --cellpose-sweep")
    parser.add_argument("--cellpose-sweep", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--hough-sweep", action="store_true")
    parser.add_argument("--preprocess-sweep", action="store_true")
    parser.add_argument("--dog-clahe-sweep", action="store_true")
    parser.add_argument("--clahe-sweep", action="store_true")
    args = parser.parse_args()

    main(
        Path(args.input),
        args.frame,
        Path(args.output_dir),
        args.bubble_smooth_sigma,
        args.cellpose_sweep or args.sweep,
        args.threshold_sweep,
        args.hough_sweep,
        args.preprocess_sweep,
        args.dog_clahe_sweep,
        args.clahe_sweep,
    )
