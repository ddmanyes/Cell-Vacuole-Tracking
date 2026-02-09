"""
Draw cell/bubble overlays from saved intermediate npz without recomputation.

Usage:
    uv run src/tests/overlay_from_npz.py \
        --npz results/intermediates/Group 1_wellA1_RI_MIP_stitched_intermediate.npz \
        --input data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff \
        --frame 0
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    min_val = frame.min()
    max_val = frame.max()
    if max_val > min_val:
        return (frame - min_val) / (max_val - min_val)
    return np.zeros_like(frame)


def build_overlay(img: np.ndarray, mask: np.ndarray, bubble_labels: np.ndarray | None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    cell_outline = find_boundaries(mask, mode="outer")
    cell_outline = dilation(cell_outline, disk(1))
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = cell_outline.astype(np.float32)
    ax.imshow(overlay)

    if bubble_labels is not None and bubble_labels.size > 0:
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


def main(npz_path: Path, input_path: Path, frame_index: int, output_dir: Path) -> None:
    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")
    if not input_path.exists():
        raise SystemExit(f"Input TIFF not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        masks = data["masks"]
        bubbles = data["bubbles"]
        bubble_method = data.get("bubble_method", np.array("unknown"))
        if isinstance(bubble_method, np.ndarray) and bubble_method.size == 1:
            bubble_method = bubble_method.item()

    with tiff.TiffFile(input_path) as tf:
        imgs = tf.asarray()

    if masks.shape[0] != imgs.shape[0]:
        raise SystemExit(
            "NPZ frames do not match TIFF frames. "
            f"npz={masks.shape[0]} tiff={imgs.shape[0]}"
        )
    if bubbles is not None and bubbles.shape != masks.shape:
        raise SystemExit(
            "NPZ bubbles do not match masks shape. "
            f"bubbles={bubbles.shape} masks={masks.shape}"
        )

    frame_index = max(0, min(frame_index, masks.shape[0] - 1))
    frame = normalize_frame(imgs[frame_index])
    mask = masks[frame_index]

    bubbles_frame = None
    if bubbles is not None and bubbles.size > 0:
        bubbles_frame = bubbles[frame_index]

    overlay = build_overlay(frame, mask, bubbles_frame)
    out_path = output_dir / f"frame_{frame_index}_overlay_{str(bubble_method)}.png"
    plt.imsave(out_path, overlay)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay from saved intermediates")
    parser.add_argument("--npz", type=str, required=True, help="Intermediate npz path")
    parser.add_argument("--input", type=str, required=True, help="Original TIFF path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--output-dir", type=str, default="results/intermediate_overlays")
    args = parser.parse_args()

    main(Path(args.npz), Path(args.input), args.frame, Path(args.output_dir))
