"""
Plot the trajectory of a single tracked cell over time.

Usage:
    uv run src/tests/track_single_cell.py --tracking-csv results/tracking_positions.csv --cell-id 12
    uv run src/tests/track_single_cell.py --tracking-csv results/tracking_positions.csv --cell-id 12 \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff --frame 0
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        return (frame - min_val) / (max_val - min_val)
    return np.zeros_like(frame)


def plot_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(df["x"], df["y"], "-o", markersize=2, linewidth=1)
    ax.set_title(f"tracked_id {int(df['tracked_id'].iloc[0])} trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_overlay(df: pd.DataFrame, frame_img: np.ndarray, out_path: Path, frame_index: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame_img, cmap="gray")
    ax.plot(df["x"], df["y"], "-o", color="red", markersize=2, linewidth=1)
    ax.set_title(f"tracked_id {int(df['tracked_id'].iloc[0])} on frame {frame_index}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(
    tracking_csv: Path,
    cell_id: int,
    input_path: Path | None,
    frame_index: int | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tracking_csv.exists():
        raise SystemExit(f"Tracking CSV not found: {tracking_csv}")

    df = pd.read_csv(tracking_csv)
    df = df[df["tracked_id"] == cell_id].sort_values("frame")
    if df.empty:
        raise SystemExit(f"No rows found for tracked_id {cell_id}")

    plot_trajectory(df, output_dir / f"tracked_{cell_id}_trajectory.png")

    if input_path is None:
        return

    if not input_path.exists():
        raise SystemExit(f"Input TIFF not found: {input_path}")

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    if frame_index is None:
        frame_index = int(df["frame"].iloc[0])

    frame_index = max(0, min(frame_index, imgs.shape[0] - 1))
    frame_img = normalize_frame(imgs[frame_index])
    plot_overlay(df, frame_img, output_dir / f"tracked_{cell_id}_frame_{frame_index}.png", frame_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot single-cell tracking trajectory")
    parser.add_argument("--tracking-csv", type=str, default="results/tracking_positions.csv")
    parser.add_argument("--cell-id", type=int, required=True)
    parser.add_argument("--input", type=str, default=None, help="Optional TIFF to draw background")
    parser.add_argument("--frame", type=int, default=None, help="Frame index for overlay")
    parser.add_argument("--output-dir", type=str, default="results/tracking_qc")
    args = parser.parse_args()

    main(
        Path(args.tracking_csv),
        args.cell_id,
        Path(args.input) if args.input else None,
        args.frame,
        Path(args.output_dir),
    )
