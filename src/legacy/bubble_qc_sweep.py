"""
Generate QC overlays for different bubble detection parameters.

Usage:
    uv run src/legacy/bubble_qc_sweep.py --input <tiff_path> --frame 0
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from skimage.feature import blob_log
from skimage.segmentation import find_boundaries
from skimage.restoration import rolling_ball
from skimage.draw import circle_perimeter

from cellpose import models


DEFAULT_INPUT = Path("data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff")
DEFAULT_OUTDIR = Path("results/bubble_qc_sweep")

CELLPOSE_MODEL_TYPE = "cyto3"
CELLPOSE_DIAMETER = 100
CELLPOSE_CELLPROB_THRESHOLD = 0.6
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_MIN_SIZE = 1200
CELLPOSE_RB_RADIUS = 50
CELLPOSE_USE_CLAHE = True
CELLPOSE_FILL_HOLES_AREA = 500
CELLPOSE_CLOSING_DISK = 4


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
    from skimage.morphology import remove_small_holes, binary_closing, disk

    processed = mask.copy()
    for label_id in np.unique(processed):
        if label_id == 0:
            continue
        region = processed == label_id
        if CELLPOSE_FILL_HOLES_AREA > 0:
            region = remove_small_holes(region, area_threshold=CELLPOSE_FILL_HOLES_AREA)
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

    if isinstance(result, tuple):
        masks = result[0]
    else:
        masks = result

    return postprocess_masks(masks.astype(np.int32))


def detect_bubbles_in_cell(cell_img, min_sigma, max_sigma, num_sigma, threshold):
    cell_img = normalize_frame(cell_img)
    if cell_img.max() == 0:
        return np.array([]).reshape(0, 3)

    ys, xs = np.where(cell_img > 0)
    if len(ys) == 0:
        return np.array([]).reshape(0, 3)

    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    cropped = cell_img[y_min:y_max, x_min:x_max]

    blobs = blob_log(
        cropped,
        min_sigma=min_sigma / 10.0,
        max_sigma=max_sigma / 10.0,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=0.5,
    )

    if len(blobs) > 0:
        blobs[:, 0] += y_min
        blobs[:, 1] += x_min

    return blobs


def detect_bubbles_frame(mask, img, min_sigma, max_sigma, num_sigma, threshold):
    blobs_all = []
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        cell_region = mask == label_id
        cell_img = img * cell_region
        blobs = detect_bubbles_in_cell(cell_img, min_sigma, max_sigma, num_sigma, threshold)
        if len(blobs) > 0:
            blobs_all.append(blobs)

    if not blobs_all:
        return np.array([]).reshape(0, 3)

    return np.vstack(blobs_all)


def qc_overlay(img, mask, blobs, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    boundaries = find_boundaries(mask, mode="outer")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = boundaries.astype(np.float32)
    ax.imshow(overlay)

    bubble_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for y, x, sigma in blobs:
        radius = max(1, int(round(sigma * np.sqrt(2))))
        rr, cc = circle_perimeter(int(round(y)), int(round(x)), radius, shape=mask.shape)
        bubble_overlay[rr, cc, 0] = 1.0
        bubble_overlay[rr, cc, 1] = 1.0
        bubble_overlay[rr, cc, 2] = 1.0
        bubble_overlay[rr, cc, 3] = 1.0
    ax.imshow(bubble_overlay)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main(input_path, frame_index, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame = imgs[frame_index]

    pre = preprocess_cellpose_frame(frame)
    mask = segment_cellpose(pre)

    param_sets = [
        (1, 6, 3, 0.01),
        (1, 8, 3, 0.02),
        (2, 8, 3, 0.02),
        (2, 10, 3, 0.03),
        (3, 12, 4, 0.04),
    ]

    for min_sigma, max_sigma, num_sigma, threshold in param_sets:
        blobs = detect_bubbles_frame(mask, frame, min_sigma, max_sigma, num_sigma, threshold)
        title = f"min{min_sigma}_max{max_sigma}_n{num_sigma}_t{threshold}"
        fig = qc_overlay(frame, mask, blobs, title)
        out_png = output_dir / f"bubble_qc_f{frame_index}_{title}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)

    print(f"Saved {len(param_sets)} bubble QC overlays to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble QC parameter sweep")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    main(Path(args.input), args.frame, Path(args.output_dir))
