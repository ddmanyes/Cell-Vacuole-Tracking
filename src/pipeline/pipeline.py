"""
Cell Vacuole Tracking Pipeline (Simple Segmentation)

This script performs:
1. TIFF time-lapse image loading
2. Cell segmentation using threshold + watershed
3. Cell tracking using LapTrack (optional)
4. Bubble/vacuole detection within each cell
5. CSV output with cell and bubble statistics
6. QC overlay images

Usage:
    uv run src/pipeline/pipeline.py --input <tiff_path> [--sample-frame N]
        [--skip-tracking] [--max-frames N]
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from laptrack import LapTrack
from skimage.measure import regionprops, label as label_img
from skimage.feature import blob_log, peak_local_max
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, binary_opening, disk, white_tophat
from skimage.segmentation import watershed, find_boundaries, expand_labels
from skimage.draw import circle_perimeter
from skimage.restoration import rolling_ball
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from cellpose import models
except Exception:
    models = None

import yaml
import sys

# ============================================================================
# Configuration (Loaded from YAML)
# ============================================================================

# Default Configuration Dictionary
CONFIG = {
    'input': {'tiff_path': 'data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff'},
    'output': {
        'results_dir': 'results',
        'qc_dir': 'results/qc',
        'intermediate_dir': 'results/intermediates'
    },
    'segmentation': {
        'method': 'cellpose',
        'min_cell_area': 200,
        'gaussian_sigma': 1.0,
        'peak_footprint': 7,
        'closing_disk': 5,
        'label_expand_pixels': 3,
        'target_coverage': 0.7,
        'max_expand_iter': 25
    },
    'cellpose': {
        'model_type': 'cyto3',
        'diameter': 100,
        'cellprob_threshold': 0.6,
        'flow_threshold': 0.4,
        'min_size': 0,
        'use_clahe': True,
        'bg_subtract': 'rolling_ball',
        'rb_radius': 50,
        'fill_holes_area': 0,
        'closing_disk': 4
    },
    'bubble': {
        'method': 'rb_clahe',
        'min_sigma': 2,
        'max_sigma': 15,
        'threshold': 0.03,
        'num_sigma': 3,
        'tophat_radius': 2,
        'smooth_sigma': 0.6,
        'min_area': 3,
        'max_area': 120,
        'ws_smooth_sigma': 1.0,
        'ws_intensity_threshold': 0.25,
        'ws_marker_quantile': 0.25,
        'cellpose_model_type': 'cyto3',
        'cellpose_diameter': 10,
        'cellpose_cellprob_threshold': 0.0,
        'cellpose_flow_threshold': 0.4,
        'cellpose_min_size': 0,
        'cellpose_smooth_sigma': 1.0,
        'th_thresh': 0.28,
        'th_min_area': 20,
        'th_max_area': None,
        'th_min_circularity': 0.1,
        'th_clahe_clip': 0.06,
        'th_rb_radius': 50,
        'qc_min_sigma': 1,
        'qc_max_sigma': 8,
        'qc_threshold': 0.02
    }
}

def load_config(config_path='config/pipeline_params.yaml'):
    """Load configuration from YAML file and update defaults."""
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return

    try:
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                update_recursive(CONFIG, yaml_config)
        print(f"Configuration loaded from {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")

def update_recursive(d, u):
    """Recursively update dictionary d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Load config initially
load_config()

# Map Config to Global Variables for compatibility
DEFAULT_INPUT = Path(CONFIG['input']['tiff_path'])
RESULTS_DIR = Path(CONFIG['output']['results_dir'])
QC_DIR = Path(CONFIG['output']['qc_dir'])
INTERMEDIATE_DIR = Path(CONFIG['output'].get('intermediate_dir', 'results/intermediates'))

MIN_CELL_AREA = CONFIG['segmentation']['min_cell_area']
GAUSSIAN_SIGMA = CONFIG['segmentation']['gaussian_sigma']
PEAK_FOOTPRINT = CONFIG['segmentation']['peak_footprint']
CLOSING_DISK = CONFIG['segmentation']['closing_disk']
LABEL_EXPAND_PIXELS = CONFIG['segmentation']['label_expand_pixels']
TARGET_COVERAGE = CONFIG['segmentation']['target_coverage']
MAX_EXPAND_ITER = CONFIG['segmentation']['max_expand_iter']
SEGMENTATION_METHOD = CONFIG['segmentation']['method']

CELLPOSE_MODEL_TYPE = CONFIG['cellpose']['model_type']
CELLPOSE_DIAMETER = CONFIG['cellpose']['diameter']
CELLPOSE_CELLPROB_THRESHOLD = CONFIG['cellpose']['cellprob_threshold']
CELLPOSE_FLOW_THRESHOLD = CONFIG['cellpose']['flow_threshold']
CELLPOSE_MIN_SIZE = CONFIG['cellpose']['min_size']
CELLPOSE_USE_CLAHE = CONFIG['cellpose']['use_clahe']
CELLPOSE_BG_SUBTRACT = CONFIG['cellpose']['bg_subtract']
CELLPOSE_RB_RADIUS = CONFIG['cellpose']['rb_radius']
CELLPOSE_FILL_HOLES_AREA = CONFIG['cellpose']['fill_holes_area']
CELLPOSE_CLOSING_DISK = CONFIG['cellpose']['closing_disk']

BUBBLE_METHOD = CONFIG['bubble']['method']
BUBBLE_MIN_SIGMA = CONFIG['bubble']['min_sigma']
BUBBLE_MAX_SIGMA = CONFIG['bubble']['max_sigma']
BUBBLE_THRESHOLD = CONFIG['bubble']['threshold']
BUBBLE_NUM_SIGMA = CONFIG['bubble']['num_sigma']
BUBBLE_TOPHAT_RADIUS = CONFIG['bubble']['tophat_radius']
BUBBLE_SMOOTH_SIGMA = CONFIG['bubble']['smooth_sigma']
BUBBLE_MIN_AREA = CONFIG['bubble']['min_area']
BUBBLE_MAX_AREA = CONFIG['bubble']['max_area']
BUBBLE_WS_SMOOTH_SIGMA = CONFIG['bubble']['ws_smooth_sigma']
BUBBLE_WS_INTENSITY_THRESHOLD = CONFIG['bubble']['ws_intensity_threshold']
BUBBLE_WS_MARKER_QUANTILE = CONFIG['bubble']['ws_marker_quantile']
BUBBLE_CELLPOSE_MODEL_TYPE = CONFIG['bubble']['cellpose_model_type']
BUBBLE_CELLPOSE_DIAMETER = CONFIG['bubble']['cellpose_diameter']
BUBBLE_CELLPOSE_CELLPROB_THRESHOLD = CONFIG['bubble']['cellpose_cellprob_threshold']
BUBBLE_CELLPOSE_FLOW_THRESHOLD = CONFIG['bubble']['cellpose_flow_threshold']
BUBBLE_CELLPOSE_MIN_SIZE = CONFIG['bubble']['cellpose_min_size']
BUBBLE_CELLPOSE_SMOOTH_SIGMA = CONFIG['bubble']['cellpose_smooth_sigma']
BUBBLE_TH_THRESH = CONFIG['bubble']['th_thresh']
BUBBLE_TH_MIN_AREA = CONFIG['bubble']['th_min_area']
BUBBLE_TH_MAX_AREA = CONFIG['bubble']['th_max_area']
BUBBLE_TH_MIN_CIRCULARITY = CONFIG['bubble']['th_min_circularity']
BUBBLE_TH_CLAHE_CLIP = CONFIG['bubble']['th_clahe_clip']
BUBBLE_TH_RB_RADIUS = CONFIG['bubble']['th_rb_radius']
BUBBLE_TH_RB_RADIUS = CONFIG['bubble']['th_rb_radius']
BUBBLE_QC_MIN_SIGMA = CONFIG['bubble']['qc_min_sigma']
BUBBLE_QC_MAX_SIGMA = CONFIG['bubble']['qc_max_sigma']
BUBBLE_QC_THRESHOLD = CONFIG['bubble']['qc_threshold']


# ============================================================================
# Setup
# ============================================================================

def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TIFF Loading
# ============================================================================

def load_tiff(filepath):
    """Load TIFF as (T, Y, X) array."""
    with tiff.TiffFile(filepath) as tf:
        data = tf.asarray()
    return data

# ============================================================================
# Preprocessing
# ============================================================================

def normalize_frame(frame):
    """Normalize frame to [0, 1] range."""
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        frame = (frame - min_val) / (max_val - min_val)
    else:
        frame = np.zeros_like(frame)
    return frame

# ============================================================================
# Cell Segmentation (Threshold + Watershed)
# ============================================================================

def segment_cells(imgs):
    """
    Segment cells using threshold + watershed.

    Args:
        imgs: (T, Y, X) image array

    Returns:
        masks: (T, Y, X) label array with cell IDs
    """
    if SEGMENTATION_METHOD == "cellpose":
        return segment_cells_cellpose(imgs)

    masks = np.zeros_like(imgs, dtype=np.int32)

    for t in range(imgs.shape[0]):
        frame = normalize_frame(imgs[t])
        smoothed = gaussian(frame, sigma=GAUSSIAN_SIGMA)

        try:
            thresh = threshold_otsu(smoothed)
        except ValueError:
            thresh = float(np.mean(smoothed))

        binary = smoothed > thresh
        binary = remove_small_objects(binary, min_size=MIN_CELL_AREA)
        binary = binary_closing(binary, disk(CLOSING_DISK))

        distance = ndimage.distance_transform_edt(binary)
        peaks = peak_local_max(
            distance,
            footprint=np.ones((PEAK_FOOTPRINT, PEAK_FOOTPRINT)),
            labels=binary,
        )

        markers = np.zeros_like(frame, dtype=np.int32)
        for idx, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = idx

        if markers.max() == 0:
            labels = label_img(binary)
        else:
            labels = watershed(-distance, markers, mask=binary)

        # Grow labels slightly to better cover cell boundaries without merging.
        if LABEL_EXPAND_PIXELS > 0:
            labels = expand_labels(labels, distance=LABEL_EXPAND_PIXELS)

        # If coverage is too low, expand labels iteratively to include bubble-rich interiors.
        if TARGET_COVERAGE is not None:
            labels = expand_labels_to_coverage(labels, TARGET_COVERAGE, MAX_EXPAND_ITER)

        masks[t] = labels

    return masks


def expand_labels_to_coverage(labels, target_coverage, max_iter):
    if target_coverage is None:
        return labels

    for _ in range(max_iter):
        coverage = float(np.mean(labels > 0))
        if coverage >= target_coverage:
            break
        labels = expand_labels(labels, distance=1)
    return labels


def preprocess_cellpose_frame(frame):
    img = normalize_frame(frame)

    if CELLPOSE_BG_SUBTRACT == "rolling_ball":
        background = rolling_ball(img, radius=CELLPOSE_RB_RADIUS)
        img = img - background
        img = normalize_frame(img)

    if CELLPOSE_USE_CLAHE:
        img = equalize_adapthist(img, clip_limit=0.02)
        img = normalize_frame(img)

    return img


def postprocess_cellpose_masks(masks):
    if CELLPOSE_FILL_HOLES_AREA <= 0 and CELLPOSE_CLOSING_DISK <= 0:
        return masks

    processed = masks.copy()
    labels = np.unique(processed)
    for label_id in labels:
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


def segment_cells_cellpose(imgs):
    if models is None:
        raise RuntimeError("Cellpose is not available. Install cellpose or change SEGMENTATION_METHOD.")

    model = models.CellposeModel(model_type=CELLPOSE_MODEL_TYPE)
    masks = np.zeros_like(imgs, dtype=np.int32)

    for t in range(imgs.shape[0]):
        img = preprocess_cellpose_frame(imgs[t])
        result = model.eval(
            img,
            channels=[0, 0],
            diameter=CELLPOSE_DIAMETER,
            cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            flow_threshold=CELLPOSE_FLOW_THRESHOLD,
            min_size=CELLPOSE_MIN_SIZE,
        )

        if isinstance(result, tuple):
            if len(result) == 4:
                masks_t, _, _, _ = result
            elif len(result) == 3:
                masks_t, _, _ = result
            else:
                masks_t = result[0]
        else:
            masks_t = result

        masks[t] = postprocess_cellpose_masks(masks_t.astype(np.int32))

    return masks


def compute_tracking_metrics(track_df):
    if track_df.empty or 'tracked_id' not in track_df.columns:
        return {}

    if 'frame' not in track_df.columns:
        return {}

    total_rows = len(track_df)
    tracked_ids = track_df['tracked_id'].nunique()

    gaps = []
    step_dists = []

    for _, group in track_df.sort_values('frame').groupby('tracked_id'):
        frames = group['frame'].to_numpy()
        if len(frames) > 1:
            frame_gaps = np.diff(frames)
            gaps.extend(frame_gaps[frame_gaps > 1].tolist())

            if 'y' not in group.columns or 'x' not in group.columns:
                continue
            coords = group[['y', 'x']].to_numpy()
            deltas = np.diff(coords, axis=0)
            step = np.sqrt((deltas ** 2).sum(axis=1))
            step_dists.extend(step.tolist())

    gap_count = len(gaps)
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    mean_step = float(np.mean(step_dists)) if step_dists else 0.0
    max_step = float(np.max(step_dists)) if step_dists else 0.0

    return {
        'track_rows': total_rows,
        'tracked_ids': int(tracked_ids),
        'gap_count': int(gap_count),
        'mean_gap': mean_gap,
        'mean_step': mean_step,
        'max_step': max_step,
    }


def write_tracking_diagnostics(track_df, imgs, sample_frame):
    if track_df.empty:
        return

    required_cols = {'tracked_id', 'frame', 'y', 'x'}
    if not required_cols.issubset(track_df.columns):
        return

    df = track_df[['tracked_id', 'frame', 'y', 'x']].copy()
    df = df.sort_values(['tracked_id', 'frame'])

    # Track length stats
    lengths = df.groupby('tracked_id')['frame'].agg(['count', 'min', 'max'])
    lengths = lengths.rename(columns={'count': 'length', 'min': 'start_frame', 'max': 'end_frame'})
    lengths.to_csv(RESULTS_DIR / 'tracking_lengths.csv', index=True)

    # Step distances
    step_rows = []
    for track_id, group in df.groupby('tracked_id'):
        coords = group[['y', 'x']].to_numpy()
        if len(coords) < 2:
            continue
        deltas = np.diff(coords, axis=0)
        steps = np.sqrt((deltas ** 2).sum(axis=1))
        for step in steps:
            step_rows.append({'tracked_id': track_id, 'step_dist': float(step)})

    if step_rows:
        step_df = pd.DataFrame(step_rows)
        step_df.to_csv(RESULTS_DIR / 'tracking_steps.csv', index=False)

        # Step histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(step_df['step_dist'], bins=40, color='steelblue', alpha=0.85)
        ax.set_title('Tracking Step Distance Histogram')
        ax.set_xlabel('pixels')
        ax.set_ylabel('count')
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'tracking_steps_hist.png', dpi=120)
        plt.close(fig)

    # Length histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths['length'], bins=40, color='seagreen', alpha=0.85)
    ax.set_title('Tracking Length Histogram')
    ax.set_xlabel('frames')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'tracking_lengths_hist.png', dpi=120)
    plt.close(fig)

    # Sample track overlay
    sample_frame = min(sample_frame, imgs.shape[0] - 1)
    base = normalize_frame(imgs[sample_frame])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base, cmap='gray')

    unique_ids = lengths.sort_values('length', ascending=False).head(20).index.tolist()
    for track_id in unique_ids:
        track = df[df['tracked_id'] == track_id]
        ax.plot(track['x'], track['y'], linewidth=1)

    ax.set_title(f'Tracking Overlay (frame {sample_frame})')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'tracking_overlay.png', dpi=120)
    plt.close(fig)


def write_tracking_positions(track_df):
    if track_df.empty:
        return

    required_cols = {'tracked_id', 'frame', 'y', 'x'}
    if not required_cols.issubset(track_df.columns):
        return

    cols = ['tracked_id', 'frame', 'y', 'x']
    if 'label' in track_df.columns:
        cols.append('label')

    out_path = RESULTS_DIR / 'tracking_positions.csv'
    track_df[cols].sort_values(['tracked_id', 'frame']).to_csv(out_path, index=False)
    print(f"Tracking positions: {out_path}")


def normalize_track_df(track_df):
    if track_df.empty:
        return track_df

    df = track_df
    if isinstance(df.index, pd.MultiIndex) or df.index.name:
        df = df.reset_index()

    rename_map = {}
    if 'frame' not in df.columns:
        for candidate in ['t', 'frame_id', 'time', 'timestep']:
            if candidate in df.columns:
                rename_map[candidate] = 'frame'
                break

    if 'label' not in df.columns:
        for candidate in ['label_id', 'obj_id', 'object_id', 'id']:
            if candidate in df.columns:
                rename_map[candidate] = 'label'
                break

    if 'tracked_id' not in df.columns and 'track_id' in df.columns:
        rename_map['track_id'] = 'tracked_id'

    if 'y' not in df.columns:
        for candidate in ['y_px', 'y_coord', 'y_center', 'row', 'cy']:
            if candidate in df.columns:
                rename_map[candidate] = 'y'
                break

    if 'x' not in df.columns:
        for candidate in ['x_px', 'x_coord', 'x_center', 'col', 'cx']:
            if candidate in df.columns:
                rename_map[candidate] = 'x'
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'tracked_id' not in df.columns and 'label' in df.columns:
        df = df.copy()
        df['tracked_id'] = df['label']

    return df


def merge_track_labels(track_df, detections_df):
    if track_df.empty:
        return track_df

    required = {'frame', 'y', 'x', 'tracked_id'}
    if not required.issubset(track_df.columns):
        return track_df

    if 'label' in track_df.columns:
        return track_df

    det = detections_df.copy()
    for col in ['y', 'x']:
        det[col] = det[col].round(4)

    merged = track_df.copy()
    for col in ['y', 'x']:
        merged[col] = merged[col].round(4)

    merged = merged.merge(
        det[['frame', 'y', 'x', 'label']],
        on=['frame', 'y', 'x'],
        how='left',
    )

    return merged

# ============================================================================
# Cell Tracking
# ============================================================================

def track_cells(masks):
    """
    Track cells across frames using LapTrack with centroid-based matching.

    Args:
        masks: (T, Y, X) label array

    Returns:
        track_df: DataFrame with columns [frame, label, y, x, tracked_id]
    """
    frames_data = []

    for t in range(masks.shape[0]):
        frame_mask = masks[t].astype(int)

        try:
            region_props = regionprops(frame_mask)
        except TypeError as e:
            print(f"Warning: regionprops failed for frame {t}: {e}")
            continue

        for prop in region_props:
            cy, cx = prop.centroid
            frames_data.append({
                'frame': t,
                'label': prop.label,
                'y': cy,
                'x': cx,
            })

    if not frames_data:
        return pd.DataFrame(columns=['frame', 'label', 'y', 'x', 'tracked_id'])

    detections_df = pd.DataFrame(frames_data)

    lt = LapTrack(
        track_cost_cutoff=50.0,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )

    try:
        track_df, _, _ = lt.predict_dataframe(
            detections_df,
            coordinate_cols=['y', 'x']
        )
    except Exception:
        track_df = detections_df.copy()
        track_df['tracked_id'] = track_df['label']

    track_df = normalize_track_df(track_df)
    track_df = merge_track_labels(track_df, detections_df)

    if 'label' not in track_df.columns:
        track_df = track_df.copy()
        track_df['label'] = track_df['tracked_id']

    return track_df

# ============================================================================
# Bubble Detection
# ============================================================================

def detect_bubbles_in_cell(
    cell_img,
    threshold=BUBBLE_THRESHOLD,
    min_sigma=BUBBLE_MIN_SIGMA,
    max_sigma=BUBBLE_MAX_SIGMA,
    num_sigma=BUBBLE_NUM_SIGMA,
):
    """
    Detect bubbles/vacuoles in a cell ROI using blob detection.

    Args:
        cell_img: 2D numpy array (cell region, outside is 0)
        threshold: blob detection threshold

    Returns:
        blobs: Nx3 array [y, x, sigma]
    """
    cell_img = normalize_frame(cell_img)

    if cell_img.max() == 0:
        return np.array([]).reshape(0, 3)

    # Crop to actual cell bounding box to avoid processing large empty regions
    ys, xs = np.where(cell_img > 0)
    if len(ys) == 0:
        return np.array([]).reshape(0, 3)
    
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    cropped = cell_img[y_min:y_max, x_min:x_max]

    try:
        blobs = blob_log(
            cropped,
            min_sigma=min_sigma / 10.0,
            max_sigma=max_sigma / 10.0,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=0.5
        )
        # Adjust blob coordinates back to original image space
        if len(blobs) > 0:
            blobs[:, 0] += y_min
            blobs[:, 1] += x_min
    except Exception:
        blobs = np.array([]).reshape(0, 3)

    return blobs


def preprocess_bubble_base(frame):
    img = preprocess_cellpose_frame(frame)
    return normalize_frame(img)


def preprocess_rb_clahe(frame):
    img = normalize_frame(frame)
    if BUBBLE_TH_RB_RADIUS and BUBBLE_TH_RB_RADIUS > 0:
        background = rolling_ball(img, radius=BUBBLE_TH_RB_RADIUS)
        img = img - background
        img = normalize_frame(img)
    img = equalize_adapthist(img, clip_limit=BUBBLE_TH_CLAHE_CLIP)
    return normalize_frame(img)


def preprocess_bubble_frame(frame):
    img = preprocess_bubble_base(frame)
    inv = 1.0 - img
    inv = white_tophat(inv, disk(BUBBLE_TOPHAT_RADIUS))
    inv = gaussian(inv, sigma=BUBBLE_SMOOTH_SIGMA)
    return normalize_frame(inv)


def filter_bubble_labels(labels, min_area, max_area):
    if labels.max() == 0:
        return labels

    cleaned = np.zeros_like(labels)
    idx = 1
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        area = int(np.sum(labels == label_id))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        cleaned[labels == label_id] = idx
        idx += 1

    return cleaned


def detect_bubbles_tophat(frame_mask, frame_img, *, use_qc_params=False):
    img = preprocess_bubble_frame(frame_img)
    if img.max() == 0:
        return np.zeros_like(frame_mask, dtype=np.int32)

    threshold = BUBBLE_QC_THRESHOLD if use_qc_params else BUBBLE_THRESHOLD
    binary = img > threshold
    binary &= (frame_mask > 0)
    binary = remove_small_objects(binary, min_size=BUBBLE_MIN_AREA)

    labels = ndimage.label(binary)[0]
    labels = filter_bubble_labels(labels, BUBBLE_MIN_AREA, BUBBLE_MAX_AREA)
    return labels


def detect_bubbles_gradient_ws(frame_mask, frame_img, *, use_qc_params=False):
    img = preprocess_bubble_base(frame_img)
    if BUBBLE_CELLPOSE_SMOOTH_SIGMA and BUBBLE_CELLPOSE_SMOOTH_SIGMA > 0:
        img = gaussian(img, sigma=BUBBLE_CELLPOSE_SMOOTH_SIGMA)
    img = normalize_frame(1.0 - img)
    if img.max() == 0:
        return np.zeros_like(frame_mask, dtype=np.int32)

    smooth = gaussian(img, sigma=BUBBLE_WS_SMOOTH_SIGMA)
    gradient = sobel(smooth)

    threshold = BUBBLE_WS_INTENSITY_THRESHOLD
    cell_vals = smooth[frame_mask > 0]
    if cell_vals.size > 0 and BUBBLE_WS_MARKER_QUANTILE is not None:
        threshold = float(np.quantile(cell_vals, BUBBLE_WS_MARKER_QUANTILE))

    markers = smooth < threshold
    markers &= (frame_mask > 0)
    markers = remove_small_objects(markers, min_size=BUBBLE_MIN_AREA)
    marker_labels = ndimage.label(markers)[0]
    if marker_labels.max() == 0:
        return np.zeros_like(frame_mask, dtype=np.int32)

    labels = watershed(gradient, marker_labels, mask=(frame_mask > 0))
    labels = filter_bubble_labels(labels, BUBBLE_MIN_AREA, BUBBLE_MAX_AREA)
    return labels


def detect_bubbles_rb_clahe(frame_mask, frame_img, *, use_qc_params=False):
    pre = preprocess_rb_clahe(frame_img)

    thresh = BUBBLE_TH_THRESH
    min_area = BUBBLE_TH_MIN_AREA
    max_area = BUBBLE_TH_MAX_AREA
    min_circularity = BUBBLE_TH_MIN_CIRCULARITY

    filtered = np.zeros_like(frame_mask, dtype=np.int32)
    next_label = 1

    for label_id in np.unique(frame_mask):
        if label_id == 0:
            continue
        cell_region = frame_mask == label_id
        ys, xs = np.where(cell_region)
        if len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        pre_roi = pre[y_min:y_max, x_min:x_max]
        cell_roi = cell_region[y_min:y_max, x_min:x_max]

        binary = (pre_roi < thresh) & cell_roi
        binary = binary_opening(binary, disk(1))
        binary = remove_small_objects(binary, min_size=min_area)

        labels = ndimage.label(binary)[0]
        if labels.max() == 0:
            continue

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

            mask = labels == prop.label
            sub = filtered[y_min:y_max, x_min:x_max]
            sub[mask] = next_label
            filtered[y_min:y_max, x_min:x_max] = sub
            next_label += 1

    return filtered


def detect_bubbles_cellpose(frame_mask, frame_img):
    if models is None:
        raise RuntimeError("Cellpose is not available. Install cellpose or change BUBBLE_METHOD.")

    img = preprocess_bubble_base(frame_img)
    model = models.CellposeModel(model_type=BUBBLE_CELLPOSE_MODEL_TYPE)

    bubble_labels = np.zeros_like(frame_mask, dtype=np.int32)
    next_label = 1

    for label_id in np.unique(frame_mask):
        if label_id == 0:
            continue
        cell_region = frame_mask == label_id
        ys, xs = np.where(cell_region)
        if len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        cell_img = img[y_min:y_max, x_min:x_max]
        cell_mask = cell_region[y_min:y_max, x_min:x_max]
        cell_img = cell_img * cell_mask

        result = model.eval(
            cell_img,
            channels=[0, 0],
            diameter=BUBBLE_CELLPOSE_DIAMETER,
            cellprob_threshold=BUBBLE_CELLPOSE_CELLPROB_THRESHOLD,
            flow_threshold=BUBBLE_CELLPOSE_FLOW_THRESHOLD,
            min_size=BUBBLE_CELLPOSE_MIN_SIZE,
        )

        if isinstance(result, tuple):
            masks = result[0]
        else:
            masks = result

        masks = masks.astype(np.int32)
        masks *= cell_mask.astype(np.int32)
        if masks.max() == 0:
            continue

        for bubble_id in np.unique(masks):
            if bubble_id == 0:
                continue
            bubble_labels[y_min:y_max, x_min:x_max][masks == bubble_id] = next_label
            next_label += 1

    bubble_labels = filter_bubble_labels(bubble_labels, BUBBLE_MIN_AREA, BUBBLE_MAX_AREA)
    return bubble_labels


def analyze_bubbles_in_frame(frame_mask, frame_img, *, return_labels=False):
    """
    Detect bubbles in all cells of a frame.

    Args:
        frame_mask: 2D label array
        frame_img: 2D image array

    Returns:
        results: list of dicts with [label, bubble_count, bubble_area, ...]
    """
    results = []

    bubble_labels = None
    bubble_owner = {}
    bubble_area_map = {}

    if BUBBLE_METHOD == "tophat":
        bubble_labels = detect_bubbles_tophat(frame_mask, frame_img)
    elif BUBBLE_METHOD == "gradient_ws":
        bubble_labels = detect_bubbles_gradient_ws(frame_mask, frame_img)
    elif BUBBLE_METHOD == "cellpose":
        bubble_labels = detect_bubbles_cellpose(frame_mask, frame_img)
    elif BUBBLE_METHOD == "rb_clahe":
        bubble_labels = detect_bubbles_rb_clahe(frame_mask, frame_img)

    if bubble_labels is not None:
        bubble_ids = np.unique(bubble_labels)
        bubble_ids = bubble_ids[bubble_ids != 0]

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

    for label_id in np.unique(frame_mask):
        if label_id == 0:
            continue

        cell_region = (frame_mask == label_id)
        cell_size = int(np.sum(cell_region))

        if BUBBLE_METHOD in {"tophat", "gradient_ws", "cellpose", "rb_clahe"}:
            owned = [b_id for b_id, owner in bubble_owner.items() if owner == label_id]
            bubble_count = int(len(owned))
            bubble_area = float(np.sum([bubble_area_map[b_id] for b_id in owned]))
            bubble_area_method = "pixel_count"
        else:
            cell_img = frame_img * cell_region
            blobs = detect_bubbles_in_cell(cell_img)
            bubble_count = len(blobs)
            bubble_area = 0.0
            if bubble_count > 0:
                radii = blobs[:, 2] * np.sqrt(2)
                bubble_area = float(np.sum(np.pi * (radii ** 2)))
            bubble_area_method = "sigma_circle"

        results.append({
            'label': int(label_id),
            'cell_size_px': cell_size,
            'bubble_count': int(bubble_count),
            'bubble_area': bubble_area,
            'bubble_area_method': bubble_area_method,
            'bubble_density': float(bubble_count / cell_size) if cell_size > 0 else 0.0,
        })

    if return_labels:
        if bubble_labels is None:
            bubble_labels = np.zeros_like(frame_mask, dtype=np.int32)
        return results, bubble_labels

    return results


def detect_bubbles_in_frame(frame_mask, frame_img, *, use_qc_params=False):
    if BUBBLE_METHOD == "tophat":
        return detect_bubbles_tophat(frame_mask, frame_img, use_qc_params=use_qc_params)
    if BUBBLE_METHOD == "gradient_ws":
        return detect_bubbles_gradient_ws(frame_mask, frame_img, use_qc_params=use_qc_params)
    if BUBBLE_METHOD == "cellpose":
        return detect_bubbles_cellpose(frame_mask, frame_img)
    if BUBBLE_METHOD == "rb_clahe":
        return detect_bubbles_rb_clahe(frame_mask, frame_img, use_qc_params=use_qc_params)

    blobs_all = []
    for label_id in np.unique(frame_mask):
        if label_id == 0:
            continue
        cell_region = (frame_mask == label_id)
        cell_img = frame_img * cell_region
        if use_qc_params:
            blobs = detect_bubbles_in_cell(
                cell_img,
                threshold=BUBBLE_QC_THRESHOLD,
                min_sigma=BUBBLE_QC_MIN_SIGMA,
                max_sigma=BUBBLE_QC_MAX_SIGMA,
                num_sigma=BUBBLE_NUM_SIGMA,
            )
        else:
            blobs = detect_bubbles_in_cell(cell_img)
        if len(blobs) > 0:
            blobs_all.append(blobs)

    if not blobs_all:
        return np.array([]).reshape(0, 3)

    return np.vstack(blobs_all)

# ============================================================================
# Analysis & CSV Output
# ============================================================================

def analyze_file(
    filepath,
    sample_frame=None,
    skip_tracking=False,
    max_frames=None,
    save_intermediates=False,
    intermediate_dir=None,
):
    """
    Complete analysis pipeline for a single TIFF file.

    Args:
        filepath: Path to TIFF
        sample_frame: Frame index for QC overlay (default: middle frame)
        skip_tracking: If True, skip LapTrack and use per-frame labels
        max_frames: If set, only process the first N frames

    Returns:
        results_df: DataFrame with all results
        qc_img: Overlay image for visualization
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")

    imgs = load_tiff(filepath)
    print(f"Loaded: shape={imgs.shape}, dtype={imgs.dtype}")

    if max_frames is not None:
        imgs = imgs[:max_frames]
        print(f"Using first {len(imgs)} frames for debug.")

    if SEGMENTATION_METHOD == "cellpose":
        print("Segmenting cells with Cellpose...")
    else:
        print("Segmenting cells with threshold + watershed...")
    masks = segment_cells(imgs)

    if skip_tracking:
        print("Tracking skipped (using per-frame labels).")
        track_df = pd.DataFrame()
    else:
        print("Tracking cells with LapTrack...")
        track_df = track_cells(masks)

    tracking_metrics = {}
    if not skip_tracking:
        tracking_metrics = compute_tracking_metrics(track_df)
        if tracking_metrics:
            tracking_path = RESULTS_DIR / 'tracking_summary.csv'
            pd.DataFrame([tracking_metrics]).to_csv(tracking_path, index=False)
            print(f"Tracking summary: {tracking_path}")
        write_tracking_positions(track_df)
        write_tracking_diagnostics(track_df, imgs, sample_frame if sample_frame is not None else 0)

    id_mapping = {}
    if not track_df.empty and {'frame', 'label', 'tracked_id'}.issubset(track_df.columns):
        for _, row in track_df.iterrows():
            frame = int(row['frame'])
            label = int(row['label'])
            tracked_id = int(row['tracked_id'])
            if frame not in id_mapping:
                id_mapping[frame] = {}
            id_mapping[frame][label] = tracked_id

    print("Detecting bubbles in cells...")
    all_results = []
    bubble_labels_all = [] if save_intermediates else None

    for t in tqdm(range(len(imgs)), desc="Frame analysis"):
        frame_mask = masks[t]
        frame_img = imgs[t]

        if save_intermediates:
            bubble_results, bubble_labels = analyze_bubbles_in_frame(
                frame_mask,
                frame_img,
                return_labels=True,
            )
            bubble_labels_all.append(bubble_labels.astype(np.int32))
        else:
            bubble_results = analyze_bubbles_in_frame(frame_mask, frame_img)

        for bubble_result in bubble_results:
            label = bubble_result['label']
            tracked_id = id_mapping.get(t, {}).get(label, label)

            all_results.append({
                'frame': t,
                'cell_id': tracked_id,
                **bubble_result,
            })

    results_df = pd.DataFrame(all_results)

    if save_intermediates:
        if intermediate_dir is None:
            intermediate_dir = INTERMEDIATE_DIR
        intermediate_dir = Path(intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        bubble_stack = np.stack(bubble_labels_all, axis=0) if bubble_labels_all else None
        out_path = intermediate_dir / f"{Path(filepath).stem}_intermediate.npz"
        np.savez_compressed(
            out_path,
            masks=masks.astype(np.int32),
            bubbles=bubble_stack.astype(np.int32) if bubble_stack is not None else None,
            bubble_method=np.array(BUBBLE_METHOD),
        )
        print(f"Saved intermediates: {out_path}")

    if sample_frame is None:
        sample_frame = len(imgs) // 2

    sample_frame = min(sample_frame, len(imgs) - 1)
    bubble_result = detect_bubbles_in_frame(
        masks[sample_frame],
        imgs[sample_frame],
        use_qc_params=True,
    )
    qc_img = create_qc_overlay(
        normalize_frame(imgs[sample_frame]),
        masks[sample_frame],
        sample_frame,
        bubble_result
    )

    print(f"Analysis complete. {len(results_df)} cell-frame records.")

    return results_df, qc_img


def create_qc_overlay(img, mask, frame_idx, bubble_blobs=None):
    """Create QC overlay with image + mask + bubble outlines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.imshow(img, cmap='gray')
    boundaries = find_boundaries(mask, mode='outer')
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = boundaries.astype(np.float32)
    ax.imshow(overlay)

    if bubble_blobs is not None:
        bubble_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        if bubble_blobs.ndim == 2 and bubble_blobs.shape[1] == 3:
            for y, x, sigma in bubble_blobs:
                radius = max(1, int(round(sigma * np.sqrt(2))))
                rr, cc = circle_perimeter(int(round(y)), int(round(x)), radius, shape=mask.shape)
                bubble_overlay[rr, cc, 0] = 1.0
                bubble_overlay[rr, cc, 1] = 1.0
                bubble_overlay[rr, cc, 2] = 1.0
                bubble_overlay[rr, cc, 3] = 1.0
        else:
            bubble_outline = find_boundaries(bubble_blobs.astype(np.int32), mode='outer')
            bubble_overlay[..., 0] = bubble_outline.astype(np.float32)
            bubble_overlay[..., 1] = bubble_outline.astype(np.float32)
            bubble_overlay[..., 2] = bubble_outline.astype(np.float32)
            bubble_overlay[..., 3] = bubble_outline.astype(np.float32)

        if bubble_overlay[..., 3].max() > 0:
            ax.imshow(bubble_overlay)

    ax.set_title(f'Frame {frame_idx}: Cells')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(img, cmap='gray', alpha=0.5)

    mask_rgb = np.zeros((*mask.shape, 3))
    unique_labels = np.unique(mask)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label_id in enumerate(unique_labels):
        if label_id == 0:
            continue
        mask_rgb[mask == label_id] = colors[i % len(colors)][:3]

    ax.imshow(mask_rgb)
    ax.set_title(f'Frame {frame_idx}: Cell Mask')
    ax.axis('off')

    plt.tight_layout()

    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    from PIL import Image
    qc_img = Image.open(buf)
    return qc_img

# ============================================================================
# Main
# ============================================================================

def main(
    input_path,
    sample_frame=None,
    skip_tracking=False,
    max_frames=None,
    save_intermediates=False,
    intermediate_dir=None,
):
    """Run pipeline on a single file."""
    setup_dirs()

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return

    results_df, qc_img = analyze_file(
        input_path,
        sample_frame=sample_frame,
        skip_tracking=skip_tracking,
        max_frames=max_frames,
        save_intermediates=save_intermediates,
        intermediate_dir=intermediate_dir,
    )

    csv_name = input_path.stem + '.csv'
    csv_path = RESULTS_DIR / csv_name
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    qc_name = input_path.stem + '_qc.png'
    qc_path = QC_DIR / qc_name
    qc_img.save(qc_path, dpi=(100, 100))
    print(f"Saved: {qc_path}")

    method_counts = {}
    if not results_df.empty and 'bubble_area_method' in results_df.columns:
        method_counts = results_df['bubble_area_method'].value_counts().to_dict()

    summary_df = pd.DataFrame([{
        'file': input_path.name,
        'total_frames': results_df['frame'].max() + 1 if not results_df.empty else 0,
        'unique_cells': results_df['cell_id'].nunique() if not results_df.empty else 0,
        'avg_bubbles_per_cell': results_df['bubble_count'].mean() if not results_df.empty else 0,
        'avg_cell_size': results_df['cell_size_px'].mean() if not results_df.empty else 0,
        'bubble_area_method_pixel_count_rows': int(method_counts.get('pixel_count', 0)),
        'bubble_area_method_sigma_circle_rows': int(method_counts.get('sigma_circle', 0)),
    }])

    summary_path = RESULTS_DIR / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))

    if not results_df.empty:
        per_cell = results_df.copy()
        per_cell['bubble_area_ratio'] = np.where(
            per_cell['bubble_area_method'] == 'pixel_count',
            per_cell['bubble_area'] / per_cell['cell_size_px'],
            np.nan,
        )
        cell_summary = per_cell.groupby('cell_id').agg(
            frames=('frame', 'count'),
            mean_bubble_count=('bubble_count', 'mean'),
            max_bubble_count=('bubble_count', 'max'),
            mean_bubble_area_ratio=('bubble_area_ratio', 'mean'),
            max_bubble_area_ratio=('bubble_area_ratio', 'max'),
            mean_cell_size=('cell_size_px', 'mean'),
        ).reset_index()

        cell_summary_path = RESULTS_DIR / 'cell_summary.csv'
        cell_summary.to_csv(cell_summary_path, index=False)
        print(f"Cell summary: {cell_summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell Vacuole Tracking Pipeline (Simple Segmentation)')
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT),
                        help='TIFF path for analysis')
    parser.add_argument('--sample-frame', type=int, default=None,
                        help='Frame index for QC overlay (default: middle frame)')
    parser.add_argument('--skip-tracking', action='store_true',
                        help='Skip LapTrack and use per-frame labels')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Only process the first N frames')
    parser.add_argument('--save-intermediates', action='store_true',
                        help='Save masks and bubble labels to a compressed npz file')
    parser.add_argument('--intermediate-dir', type=str, default=None,
                        help='Directory for saved intermediate npz files')
    args = parser.parse_args()

    main(
        input_path=args.input,
        sample_frame=args.sample_frame,
        skip_tracking=args.skip_tracking,
        max_frames=args.max_frames,
        save_intermediates=args.save_intermediates,
        intermediate_dir=args.intermediate_dir,
    )
