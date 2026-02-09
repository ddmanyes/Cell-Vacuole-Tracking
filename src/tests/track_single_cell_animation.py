"""
Create a per-frame animation for tracked cells.

Usage:
    # Single cell
    uv run src/tests/track_single_cell_animation.py \
        --tracking-csv results/tracking_positions.csv \
        --cell-id 12 \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff \
        --output results/tracking_qc/tracked_12.gif
    
    # Multiple cells
    uv run src/tests/track_single_cell_animation.py \
        --tracking-csv results/tracking_positions.csv \
        --cell-ids 0,1,2 \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff \
        --output results/tracking_qc/tracked_multi.gif
    
    # Top N longest tracks
    uv run src/tests/track_single_cell_animation.py \
        --tracking-csv results/tracking_positions.csv \
        --top-n 10 \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff \
        --output results/tracking_qc/tracked_top10.gif
    
    # All cells with minimum length filter
    uv run src/tests/track_single_cell_animation.py \
        --tracking-csv results/tracking_positions.csv \
        --all \
        --min-length 10 \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff \
        --output results/tracking_qc/tracked_all.gif
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        return (frame - min_val) / (max_val - min_val)
    return np.zeros_like(frame)


def build_outlines(
    mask_frame: np.ndarray,
    label_id: int | None,
    bubbles_frame: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    cell_outline = None
    bubble_outline = None

    if mask_frame is not None and label_id is not None:
        region = mask_frame == label_id
        if np.any(region):
            cell_outline = dilation(find_boundaries(region, mode="outer"), disk(2))

    if bubbles_frame is not None and bubbles_frame.size > 0:
        if mask_frame is not None and label_id is not None:
            region = mask_frame == label_id
            bubbles_region = bubbles_frame * region
        else:
            bubbles_region = bubbles_frame
        bubble_outline = dilation(find_boundaries(bubbles_region.astype(np.int32), mode="outer"), disk(1))

    return cell_outline, bubble_outline


def main(
    tracking_csv: Path,
    cell_ids: list[int],
    input_path: Path,
    output_path: Path,
    npz_path: Path | None,
    fps: int,
    show_outlines: bool = True,
) -> None:
    if not tracking_csv.exists():
        raise SystemExit(f"Tracking CSV not found: {tracking_csv}")
    if not input_path.exists():
        raise SystemExit(f"Input TIFF not found: {input_path}")

    df_all = pd.read_csv(tracking_csv)
    
    # Filter for requested cells
    df = df_all[df_all["tracked_id"].isin(cell_ids)].copy()
    if df.empty:
        raise SystemExit(f"No rows found for tracked_ids {cell_ids}")
    
    print(f"Animating {len(cell_ids)} cell(s): {cell_ids}")

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    masks = None
    bubbles = None
    if npz_path is not None and show_outlines:
        if not npz_path.exists():
            print(f"Warning: NPZ not found: {npz_path}, skipping outlines")
        else:
            with np.load(npz_path, allow_pickle=True) as data:
                masks = data.get("masks")
                bubbles = data.get("bubbles")
            if masks is not None and masks.shape[0] != imgs.shape[0]:
                print(
                    f"Warning: NPZ has {masks.shape[0]} frames but TIFF has {imgs.shape[0]} frames. "
                    f"Using only first {masks.shape[0]} frames from TIFF."
                )
                imgs = imgs[:masks.shape[0]]
                df = df[df["frame"] < masks.shape[0]]
                if df.empty:
                    raise SystemExit(f"No tracking data within first {masks.shape[0]} frames")

    # Get all frames that have at least one tracked cell
    all_frames = sorted(df["frame"].unique())
    if not all_frames:
        raise SystemExit("No frames to animate")
    
    # Prepare per-cell data
    cell_data = {}
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_ids)))
    
    for idx, cid in enumerate(cell_ids):
        cell_df = df[df["tracked_id"] == cid].sort_values("frame")
        if not cell_df.empty:
            cell_data[cid] = {
                'df': cell_df,
                'color': colors[idx],
                'coords': cell_df[["x", "y"]].to_numpy(),
                'frames': cell_df["frame"].astype(int).to_list(),
            }
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 8))
    img_artist = ax.imshow(normalize_frame(imgs[all_frames[0]]), cmap="gray")
    
    # Create trail artists for each cell
    trail_artists = {}
    for cid, data in cell_data.items():
        line, = ax.plot([], [], "-o", color=data['color'], markersize=2, linewidth=1, 
                       label=f"cell {cid}")
        trail_artists[cid] = line
    
    if len(cell_ids) <= 10:  # Only show legend for small number of cells
        ax.legend(loc='upper right', fontsize=8)
    
    # Create overlay artists
    cell_overlay = ax.imshow(np.zeros((*imgs[0].shape, 4)), alpha=1.0)
    bubble_overlay = ax.imshow(np.zeros((*imgs[0].shape, 4)), alpha=1.0)

    ax.set_title(f"Tracking {len(cell_ids)} cell(s)")
    ax.axis("off")

    has_label = "label" in df.columns
    if not has_label and show_outlines:
        print("Warning: tracking CSV has no 'label' column; cell/bubble outlines will be skipped.")

    def update(i: int):
        frame_idx = all_frames[i]
        img_artist.set_data(normalize_frame(imgs[frame_idx]))

        # Update trails for each cell
        for cid, data in cell_data.items():
            cell_df = data['df']
            # Find all positions up to current frame
            mask = cell_df['frame'] <= frame_idx
            if mask.any():
                visible_coords = data['coords'][mask]
                trail_artists[cid].set_data(visible_coords[:, 0], visible_coords[:, 1])
            else:
                trail_artists[cid].set_data([], [])
        
        # Build outlines for all cells in current frame
        cell_rgba = np.zeros((*imgs[0].shape, 4), dtype=np.float32)
        bubble_rgba = np.zeros((*imgs[0].shape, 4), dtype=np.float32)
        
        if show_outlines and masks is not None and has_label:
            frame_cells = df[df['frame'] == frame_idx]
            mask_frame = masks[frame_idx]
            bubbles_frame = bubbles[frame_idx] if bubbles is not None else None
            
            for _, row in frame_cells.iterrows():
                label_id = int(row['label'])
                cell_outline, bubble_outline = build_outlines(mask_frame, label_id, bubbles_frame)
                
                if cell_outline is not None:
                    cell_rgba[..., 2] += cell_outline.astype(np.float32)
                    cell_rgba[..., 3] += cell_outline.astype(np.float32)
                
                if bubble_outline is not None:
                    bubble_rgba[..., 0] += bubble_outline.astype(np.float32)
                    bubble_rgba[..., 1] += bubble_outline.astype(np.float32)
                    bubble_rgba[..., 2] += bubble_outline.astype(np.float32)
                    bubble_rgba[..., 3] += bubble_outline.astype(np.float32)
        
        # Clip to [0, 1] in case of overlaps
        cell_rgba = np.clip(cell_rgba, 0, 1)
        bubble_rgba = np.clip(bubble_rgba, 0, 1)
        
        cell_overlay.set_data(cell_rgba)
        bubble_overlay.set_data(bubble_rgba)

        ax.set_title(f"Tracking {len(cell_ids)} cell(s) / frame {frame_idx}")
        
        return [img_artist, cell_overlay, bubble_overlay] + list(trail_artists.values())

    anim = FuncAnimation(fig, update, frames=len(all_frames), interval=1000 / fps, blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"Saved: {output_path} ({len(all_frames)} frames)")


def select_cells(
    tracking_csv: Path,
    cell_id: int | None,
    cell_ids_str: str | None,
    top_n: int | None,
    all_cells: bool,
    min_length: int,
) -> list[int]:
    """Select which cells to animate based on arguments."""
    df = pd.read_csv(tracking_csv)
    
    # Calculate track lengths
    lengths = df.groupby('tracked_id').size().sort_values(ascending=False)
    
    # Apply min_length filter
    if min_length > 1:
        lengths = lengths[lengths >= min_length]
        print(f"Filtered to {len(lengths)} tracks with length >= {min_length}")
    
    if lengths.empty:
        raise SystemExit("No tracks found matching filters")
    
    # Select cells based on mode
    selected = []
    
    if cell_id is not None:
        selected = [cell_id]
    elif cell_ids_str is not None:
        selected = [int(x.strip()) for x in cell_ids_str.split(',')]
    elif top_n is not None:
        selected = lengths.head(top_n).index.tolist()
        print(f"Selected top {len(selected)} longest tracks")
    elif all_cells:
        selected = lengths.index.tolist()
        print(f"Selected all {len(selected)} tracks")
    else:
        raise SystemExit("Must specify one of: --cell-id, --cell-ids, --top-n, or --all")
    
    # Verify all selected cells exist
    available = set(df['tracked_id'].unique())
    missing = set(selected) - available
    if missing:
        raise SystemExit(f"Cell ID(s) not found: {sorted(missing)}")
    
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create tracking animation for one or more cells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single cell
  python track_single_cell_animation.py --cell-id 0 --input data.tiff
  
  # Multiple cells
  python track_single_cell_animation.py --cell-ids 0,1,2 --input data.tiff
  
  # Top 5 longest tracks
  python track_single_cell_animation.py --top-n 5 --input data.tiff
  
  # All cells with at least 10 frames
  python track_single_cell_animation.py --all --min-length 10 --input data.tiff
        """
    )
    parser.add_argument("--tracking-csv", type=str, default="results/tracking_positions.csv",
                       help="Path to tracking positions CSV")
    
    # Cell selection (mutually exclusive group)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell-id", type=int, help="Single cell ID to track")
    group.add_argument("--cell-ids", type=str, help="Multiple cell IDs (comma-separated, e.g., '0,1,2')")
    group.add_argument("--top-n", type=int, help="Select top N longest tracks")
    group.add_argument("--all", action="store_true", help="Animate all tracked cells")
    
    parser.add_argument("--input", type=str, required=True, help="Original TIFF path")
    parser.add_argument("--output", type=str, default="results/tracking_qc/tracked_animation.gif",
                       help="Output GIF path")
    parser.add_argument("--npz", type=str, default=None, 
                       help="Optional intermediates npz for cell/bubble outlines")
    parser.add_argument("--fps", type=int, default=6, help="Animation frames per second")
    parser.add_argument("--min-length", type=int, default=1, 
                       help="Minimum track length to include (default: 1)")
    parser.add_argument("--no-outlines", action="store_true",
                       help="Skip drawing cell/bubble outlines (faster)")
    
    args = parser.parse_args()

    # Select cells based on arguments
    selected_ids = select_cells(
        Path(args.tracking_csv),
        args.cell_id,
        args.cell_ids,
        args.top_n,
        args.all,
        args.min_length,
    )

    main(
        Path(args.tracking_csv),
        selected_ids,
        Path(args.input),
        Path(args.output),
        Path(args.npz) if args.npz else None,
        args.fps,
        show_outlines=not args.no_outlines,
    )

