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
    uv run src/pipeline.py --input <tiff_path> [--sample-frame N]
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
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.segmentation import watershed, find_boundaries
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_INPUT = Path('data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff')

RESULTS_DIR = Path('results')
QC_DIR = RESULTS_DIR / 'qc'

# Segmentation config
MIN_CELL_AREA = 200
GAUSSIAN_SIGMA = 1.0
PEAK_FOOTPRINT = 5

# Bubble detection config
BUBBLE_MIN_SIGMA = 2
BUBBLE_MAX_SIGMA = 15
BUBBLE_THRESHOLD = 0.05
BUBBLE_NUM_SIGMA = 3  # Reduced from 10 for speed

# ============================================================================
# Setup
# ============================================================================

def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    QC_DIR.mkdir(exist_ok=True)

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
        binary = binary_closing(binary, disk(3))

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

        masks[t] = labels

    return masks

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

    return track_df

# ============================================================================
# Bubble Detection
# ============================================================================

def detect_bubbles_in_cell(cell_img, threshold=BUBBLE_THRESHOLD):
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
            min_sigma=BUBBLE_MIN_SIGMA / 10.0,
            max_sigma=BUBBLE_MAX_SIGMA / 10.0,
            num_sigma=BUBBLE_NUM_SIGMA,
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


def analyze_bubbles_in_frame(frame_mask, frame_img):
    """
    Detect bubbles in all cells of a frame.

    Args:
        frame_mask: 2D label array
        frame_img: 2D image array

    Returns:
        results: list of dicts with [label, bubble_count, bubble_area, ...]
    """
    results = []

    for label_id in np.unique(frame_mask):
        if label_id == 0:
            continue

        cell_region = (frame_mask == label_id)
        cell_img = frame_img * cell_region
        cell_size = int(np.sum(cell_region))

        blobs = detect_bubbles_in_cell(cell_img)
        bubble_count = len(blobs)

        bubble_area = 0.0
        if bubble_count > 0:
            radii = blobs[:, 2] * np.sqrt(2)
            bubble_area = float(np.sum(np.pi * (radii ** 2)))

        results.append({
            'label': int(label_id),
            'cell_size_px': cell_size,
            'bubble_count': int(bubble_count),
            'bubble_area': bubble_area,
            'bubble_density': float(bubble_count / cell_size) if cell_size > 0 else 0.0,
        })

    return results

# ============================================================================
# Analysis & CSV Output
# ============================================================================

def analyze_file(filepath, sample_frame=None, skip_tracking=False, max_frames=None):
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

    print("Segmenting cells with threshold + watershed...")
    masks = segment_cells(imgs)

    if skip_tracking:
        print("Tracking skipped (using per-frame labels).")
        track_df = pd.DataFrame()
    else:
        print("Tracking cells with LapTrack...")
        track_df = track_cells(masks)

    id_mapping = {}
    if not track_df.empty:
        for _, row in track_df.iterrows():
            frame = int(row['frame'])
            label = int(row['label'])
            tracked_id = int(row['tracked_id'])
            if frame not in id_mapping:
                id_mapping[frame] = {}
            id_mapping[frame][label] = tracked_id

    print("Detecting bubbles in cells...")
    all_results = []

    for t in tqdm(range(len(imgs)), desc="Frame analysis"):
        frame_mask = masks[t]
        frame_img = imgs[t]

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

    if sample_frame is None:
        sample_frame = len(imgs) // 2

    sample_frame = min(sample_frame, len(imgs) - 1)
    qc_img = create_qc_overlay(
        normalize_frame(imgs[sample_frame]),
        masks[sample_frame],
        sample_frame
    )

    print(f"Analysis complete. {len(results_df)} cell-frame records.")

    return results_df, qc_img


def create_qc_overlay(img, mask, frame_idx):
    """Create QC overlay with image + mask."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.imshow(img, cmap='gray')
    boundaries = find_boundaries(mask, mode='outer')
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = boundaries.astype(np.float32)
    ax.imshow(overlay)

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

def main(input_path, sample_frame=None, skip_tracking=False, max_frames=None):
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
    )

    csv_name = input_path.stem + '.csv'
    csv_path = RESULTS_DIR / csv_name
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    qc_name = input_path.stem + '_qc.png'
    qc_path = QC_DIR / qc_name
    qc_img.save(qc_path, dpi=(100, 100))
    print(f"Saved: {qc_path}")

    summary_df = pd.DataFrame([{
        'file': input_path.name,
        'total_frames': results_df['frame'].max() + 1 if not results_df.empty else 0,
        'unique_cells': results_df['cell_id'].nunique() if not results_df.empty else 0,
        'avg_bubbles_per_cell': results_df['bubble_count'].mean() if not results_df.empty else 0,
        'avg_cell_size': results_df['cell_size_px'].mean() if not results_df.empty else 0,
    }])

    summary_path = RESULTS_DIR / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    print(summary_df.to_string(index=False))


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
    args = parser.parse_args()

    main(
        input_path=args.input,
        sample_frame=args.sample_frame,
        skip_tracking=args.skip_tracking,
        max_frames=args.max_frames,
    )
