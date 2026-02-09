"""
Compare bubble detection methods (blob_log vs watershed) inside Cellpose cell masks.

Usage:
    uv run src/legacy/bubble_method_compare.py --input <tiff_path> --frame 0
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import blob_log, peak_local_max
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.restoration import rolling_ball
from skimage.segmentation import watershed, find_boundaries
from skimage.draw import circle_perimeter
from scipy import ndimage


DEFAULT_INPUT = Path("data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff")
DEFAULT_OUTDIR = Path("results/bubble_method_compare")

CELLPOSE_MODEL_TYPE = "cyto3"
CELLPOSE_DIAMETER = 100
CELLPOSE_CELLPROB_THRESHOLD = 0.6
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_MIN_SIZE = 1200
CELLPOSE_RB_RADIUS = 50
CELLPOSE_USE_CLAHE = True
CELLPOSE_FILL_HOLES_AREA = 500
CELLPOSE_CLOSING_DISK = 4

# Blob_log params (QC visualization)
BLOB_MIN_SIGMA = 1
BLOB_MAX_SIGMA = 8
BLOB_NUM_SIGMA = 3
BLOB_THRESHOLD = 0.02

# Watershed bubble params
WS_SMOOTH_SIGMA = 1.0
WS_MIN_SIZE = 10
WS_MIN_DISTANCE = 2
WS_USE_INVERT = True
WS_CLAHE_CLIP = 0.0
WS_BG_SIGMA = 6.0


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

    if isinstance(result, tuple):
        masks = result[0]
    else:
        masks = result

    return postprocess_masks(masks.astype(np.int32))


def detect_bubbles_bloblog(cell_img: np.ndarray) -> np.ndarray:
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
        min_sigma=BLOB_MIN_SIGMA / 10.0,
        max_sigma=BLOB_MAX_SIGMA / 10.0,
        num_sigma=BLOB_NUM_SIGMA,
        threshold=BLOB_THRESHOLD,
        overlap=0.5,
    )

    if len(blobs) > 0:
        blobs[:, 0] += y_min
        blobs[:, 1] += x_min

    return blobs


def preprocess_bubble_img(cell_img: np.ndarray) -> np.ndarray:
    img = normalize_frame(cell_img)
    if WS_USE_INVERT:
        img = 1.0 - img

    img = gaussian(img, sigma=WS_BG_SIGMA)
    img = normalize_frame(img)
    if WS_CLAHE_CLIP > 0:
        img = equalize_adapthist(img, clip_limit=WS_CLAHE_CLIP)
        img = normalize_frame(img)
    return img


def detect_bubbles_watershed(cell_img: np.ndarray) -> np.ndarray:
    cell_img = normalize_frame(cell_img)
    if cell_img.max() == 0:
        return np.zeros_like(cell_img, dtype=np.int32)

    enhanced = preprocess_bubble_img(cell_img)
    smoothed = gaussian(enhanced, sigma=WS_SMOOTH_SIGMA)
    try:
        thresh = threshold_otsu(smoothed[smoothed > 0])
    except ValueError:
        thresh = float(np.mean(smoothed))

    binary = smoothed > thresh
    binary = remove_small_objects(binary, min_size=WS_MIN_SIZE)

    distance = ndimage.distance_transform_edt(binary)
    peaks = peak_local_max(distance, min_distance=WS_MIN_DISTANCE, labels=binary)

    markers = np.zeros_like(cell_img, dtype=np.int32)
    for idx, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = idx

    if markers.max() == 0:
        return np.zeros_like(cell_img, dtype=np.int32)

    labels = watershed(-distance, markers, mask=binary)
    return labels


def build_bubble_outline_blob(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    outline = np.zeros(mask.shape, dtype=bool)
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        cell_region = mask == label_id
        cell_img = img * cell_region
        blobs = detect_bubbles_bloblog(cell_img)
        for y, x, sigma in blobs:
            radius = max(1, int(round(sigma * np.sqrt(2))))
            rr, cc = circle_perimeter(int(round(y)), int(round(x)), radius, shape=mask.shape)
            outline[rr, cc] = True
    return outline


def build_bubble_outline_ws(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    outline = np.zeros(mask.shape, dtype=bool)
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        cell_region = mask == label_id
        cell_img = img * cell_region
        labels = detect_bubbles_watershed(cell_img)
        if labels.max() == 0:
            continue
        outline |= find_boundaries(labels, mode="outer")
    return outline


def save_overlay(img: np.ndarray, mask: np.ndarray, bubble_outline: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    cell_outline = find_boundaries(mask, mode="outer")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = cell_outline.astype(np.float32)
    ax.imshow(overlay)

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

    pre = preprocess_cellpose_frame(frame)
    mask = segment_cellpose(pre)

    bubble_outline_blob = build_bubble_outline_blob(mask, frame)
    bubble_outline_ws = build_bubble_outline_ws(mask, frame)

    save_overlay(
        frame,
        mask,
        bubble_outline_blob,
        f"frame {frame_index} / blob_log",
        output_dir / f"bubble_compare_f{frame_index}_blob_log.png",
    )

    save_overlay(
        frame,
        mask,
        bubble_outline_ws,
        f"frame {frame_index} / watershed_inv",
        output_dir / f"bubble_compare_f{frame_index}_watershed_inv.png",
    )

    print(f"Saved overlays to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble detection method comparison")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    main(Path(args.input), args.frame, Path(args.output_dir))
