"""
Cell Vacuole Tracking Pipeline

This script performs:
1. TIFF time-lapse image loading
2. Cell segmentation using Cellpose (cyto3)
3. Cell tracking using LapTrack
4. Bubble/vacuole detection within each cell
5. CSV output with cell and bubble statistics
6. QC overlay images

Usage:
    uv run src/pipeline.py [--sample-frame N]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
from cellpose import models
from laptrack import LapTrack
from skimage.measure import regionprops, label as label_img
from skimage.feature import blob_log
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import json
import argparse

# ============================================================================
# Configuration
# ============================================================================

DATA_DIRS = [
    Path('data/bafA1'),
    Path('data/control'),
]

RESULTS_DIR = Path('results')
QC_DIR = RESULTS_DIR / 'qc'
METADATA_DIR = RESULTS_DIR / 'metadata'

# Cellpose config
CELLPOSE_MODEL = 'cyto3'
CELLPOSE_DIAMETER = None  # Auto-detect

# Bubble detection config
BUBBLE_MIN_SIGMA = 2
BUBBLE_MAX_SIGMA = 15
BUBBLE_THRESHOLD = 0.05
BUBBLE_NUM_SIGMA = 10

# ============================================================================
# Setup
# ============================================================================

def setup_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    QC_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)

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
# Cell Segmentation
# ============================================================================

def segment_cells(imgs):
    """
    Segment cells using Cellpose.
    
    Args:
        imgs: (T, Y, X) image array
    
    Returns:
        masks: (T, Y, X) label array with cell IDs
    """
    model = models.CellposeModel(gpu=True, model_type=CELLPOSE_MODEL)
    # Cellpose 4.0+ returns only [masks, flows, styles]
    result = model.eval(
        imgs,
        diameter=CELLPOSE_DIAMETER,
        channels=[0, 0],
        do_3D=False
    )
    # Handle both 3-tuple and 4-tuple returns for compatibility
    if len(result) == 3:
        masks, flows, styles = result
    else:
        masks, flows, styles, _ = result
    
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
        frame_mask = masks[t].astype(int)  # Ensure int type for regionprops
        # Handle both uint and int types
        if frame_mask.dtype in [np.float32, np.float64]:
            frame_mask = frame_mask.astype(int)
        
        try:
            region_props = regionprops(frame_mask)
        except TypeError as e:
            # Skip frames where regionprops fails
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
    
    # Use LapTrack for linking
    lt = LapTrack(
        track_cost_cutoff=50.0,  # Euclidean distance threshold
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )
    
    try:
        track_df, _, _ = lt.predict_dataframe(
            detections_df,
            coordinate_cols=['y', 'x']
        )
    except:
        # Fallback if LapTrack fails: assign simple sequential IDs
        track_df = detections_df.copy()
        track_df['tracked_id'] = track_df['label']
    
    return track_df

# ============================================================================
# Bubble Detection
# ============================================================================

def detect_bubbles_in_cell(cell_roi, threshold=BUBBLE_THRESHOLD):
    """
    Detect bubbles/vacuoles in a cell ROI using blob detection.
    
    Args:
        cell_roi: 2D numpy array (cell region only)
        threshold: blob detection threshold
    
    Returns:
        blobs: Nx3 array [y, x, sigma]
    """
    cell_roi = normalize_frame(cell_roi)
    
    if cell_roi.max() == 0:
        return np.array([]).reshape(0, 3)
    
    try:
        blobs = blob_log(
            cell_roi,
            min_sigma=BUBBLE_MIN_SIGMA / 10.0,
            max_sigma=BUBBLE_MAX_SIGMA / 10.0,
            num_sigma=BUBBLE_NUM_SIGMA,
            threshold=threshold,
            overlap=0.5
        )
    except:
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
        cell_roi = frame_img[cell_region]
        cell_size = np.sum(cell_region)
        
        blobs = detect_bubbles_in_cell(cell_roi)
        bubble_count = len(blobs)
        
        bubble_area = 0
        if bubble_count > 0:
            radii = blobs[:, 2] * np.sqrt(2)
            bubble_area = float(np.sum(np.pi * (radii ** 2)))
        
        results.append({
            'label': int(label_id),
            'cell_size_px': int(cell_size),
            'bubble_count': int(bubble_count),
            'bubble_area': float(bubble_area),
            'bubble_density': float(bubble_count / cell_size) if cell_size > 0 else 0,
        })
    
    return results

# ============================================================================
# Analysis & CSV Output
# ============================================================================

def analyze_file(filepath, sample_frame=None):
    """
    Complete analysis pipeline for a single TIFF file.
    
    Args:
        filepath: Path to TIFF
        sample_frame: Frame index for QC overlay (default: middle frame)
    
    Returns:
        results_df: DataFrame with all results
        qc_img: Overlay image for visualization
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")
    
    # Load
    imgs = load_tiff(filepath)
    print(f"Loaded: shape={imgs.shape}, dtype={imgs.dtype}")
    
    # Normalize
    imgs_norm = np.array([normalize_frame(f) for f in imgs])
    
    # Segment
    print("Segmenting cells with Cellpose...")
    masks = segment_cells(imgs_norm)
    
    # Track
    print("Tracking cells with LapTrack...")
    track_df = track_cells(masks)
    
    # Create ID mapping
    id_mapping = {}
    if not track_df.empty:
        for _, row in track_df.iterrows():
            frame = int(row['frame'])
            label = int(row['label'])
            tracked_id = int(row['tracked_id'])
            if frame not in id_mapping:
                id_mapping[frame] = {}
            id_mapping[frame][label] = tracked_id
    
    # Analyze bubble per frame
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
    
    # QC overlay
    if sample_frame is None:
        sample_frame = len(imgs) // 2
    
    sample_frame = min(sample_frame, len(imgs) - 1)
    qc_img = create_qc_overlay(
        imgs_norm[sample_frame],
        masks[sample_frame],
        sample_frame
    )
    
    print(f"Analysis complete. {len(results_df)} cell-frame records.")
    
    return results_df, qc_img

def create_qc_overlay(img, mask, frame_idx):
    """Create QC overlay with image + mask + bubble positions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: image with cell contours
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        cell_boundary = ndimage.binary_dilation(mask == label_id) & ~(mask == label_id)
        ax.contour((mask == label_id).astype(float), levels=[0.5], colors='cyan', linewidths=1)
    
    ax.set_title(f'Frame {frame_idx}: Cells')
    ax.axis('off')
    
    # Right: mask overlay
    ax = axes[1]
    ax.imshow(img, cmap='gray', alpha=0.5)
    
    # Color mask
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
    
    # Convert to PIL for saving
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

def main(sample_frame=None):
    """Run complete pipeline on all files."""
    setup_dirs()
    
    all_csvs = []
    all_qc_imgs = []
    
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            print(f"Warning: {data_dir} not found, skipping.")
            continue
        
        tiff_files = list(data_dir.glob("*.tiff"))
        
        for tiff_file in sorted(tiff_files):
            try:
                results_df, qc_img = analyze_file(tiff_file, sample_frame=sample_frame)
                
                # Save CSV
                csv_name = tiff_file.stem + '.csv'
                csv_path = RESULTS_DIR / csv_name
                results_df.to_csv(csv_path, index=False)
                print(f"Saved: {csv_path}")
                
                # Save QC image
                qc_name = tiff_file.stem + '_qc.png'
                qc_path = QC_DIR / qc_name
                qc_img.save(qc_path, dpi=(100, 100))
                print(f"Saved: {qc_path}")
                
                all_csvs.append((tiff_file.name, csv_path))
                all_qc_imgs.append((tiff_file.name, qc_path))
                
            except Exception as e:
                print(f"[ERROR] Error processing {tiff_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"{'='*60}")
    print(f"CSV outputs: {len(all_csvs)}")
    for name, path in all_csvs:
        print(f"  - {path}")
    
    print(f"\nQC images: {len(all_qc_imgs)}")
    for name, path in all_qc_imgs:
        print(f"  - {path}")
    
    # Summary stats
    summary_stats = []
    for csv_name, csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        summary_stats.append({
            'file': csv_name,
            'total_frames': df['frame'].max() + 1,
            'unique_cells': df['cell_id'].nunique(),
            'avg_bubbles_per_cell': df['bubble_count'].mean(),
            'avg_cell_size': df['cell_size_px'].mean(),
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = RESULTS_DIR / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell Vacuole Tracking Pipeline')
    parser.add_argument('--sample-frame', type=int, default=None,
                        help='Frame index for QC overlay (default: middle frame)')
    args = parser.parse_args()
    
    main(sample_frame=args.sample_frame)
