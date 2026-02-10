"""Background workers for computationally-heavy pipeline tasks.

All functions decorated with ``@thread_worker`` run on a secondary
thread so that the Napari GUI stays responsive.
"""

from __future__ import annotations

import numpy as np
from napari.qt.threading import thread_worker

from src.pipeline.pipeline import (
    analyze_bubbles_in_frame,
    detect_bubbles_cellpose,
    detect_bubbles_gradient_ws,
    detect_bubbles_rb_clahe,
    detect_bubbles_tophat,
    normalize_frame,
    segment_cells,
    segment_cells_cellpose,
    track_cells,
)


# ---- Single-frame worker ---------------------------------------------------

@thread_worker
def run_single_frame(
    frame: np.ndarray,
    *,
    seg_method: str = "cellpose",
    seg_params: dict | None = None,
    bubble_method: str = "rb_clahe",
    bubble_params: dict | None = None,
):
    """Run segmentation + bubble detection on **one** frame.

    Parameters
    ----------
    frame : np.ndarray
        2-D image array (Y, X).
    seg_method : str
        ``"cellpose"`` or ``"watershed"``.
    seg_params : dict, optional
        Keyword arguments forwarded to the segmentation function.
    bubble_method : str
        ``"rb_clahe"`` | ``"tophat"`` | ``"gradient_ws"`` | ``"cellpose"``.
    bubble_params : dict, optional
        Keyword arguments forwarded to the bubble detection function.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(cell_masks, bubble_labels)`` — both 2-D label arrays.
    """
    seg_params = seg_params or {}
    bubble_params = bubble_params or {}

    # --- 1. Segment cells ---------------------------------------------------
    # Wrap single frame to (1, Y, X) as pipeline functions expect (T, Y, X).
    imgs = frame[np.newaxis, ...]

    if seg_method == "cellpose":
        masks = segment_cells_cellpose(imgs, **seg_params)
    else:
        masks = segment_cells(imgs, method="watershed", **seg_params)

    cell_mask = masks[0]  # back to (Y, X)

    # --- 2. Detect bubbles --------------------------------------------------
    norm_frame = normalize_frame(frame)

    _bubble_dispatchers = {
        "rb_clahe": detect_bubbles_rb_clahe,
        "tophat": detect_bubbles_tophat,
        "gradient_ws": detect_bubbles_gradient_ws,
        "cellpose": detect_bubbles_cellpose,
    }

    detect_fn = _bubble_dispatchers.get(bubble_method, detect_bubbles_rb_clahe)
    bubble_labels = detect_fn(cell_mask, norm_frame, **bubble_params)

    yield cell_mask, bubble_labels


# ---- Full-video worker (Phase 2 placeholder) --------------------------------

@thread_worker
def run_full_video(
    imgs: np.ndarray,
    *,
    seg_method: str = "cellpose",
    seg_params: dict | None = None,
    bubble_method: str = "rb_clahe",
    bubble_params: dict | None = None,
):
    """Process every frame — segmentation + bubble detection.

    Yields ``(frame_index, total_frames)`` tuples for progress updates,
    then yields the final results as the last item.
    """
    seg_params = seg_params or {}
    bubble_params = bubble_params or {}

    n_frames = imgs.shape[0]

    # --- 1. Segment all frames -----------------------------------------------
    if seg_method == "cellpose":
        all_masks = segment_cells_cellpose(imgs, **seg_params)
    else:
        all_masks = segment_cells(imgs, method="watershed", **seg_params)

    # --- 2. Detect bubbles per frame -----------------------------------------
    all_bubble_labels = np.zeros_like(all_masks, dtype=np.int32)
    all_results = []

    for t in range(n_frames):
        frame_mask = all_masks[t]
        frame_img = normalize_frame(imgs[t])

        results, bubble_labels = analyze_bubbles_in_frame(
            frame_mask, frame_img,
            return_labels=True,
            bubble_method=bubble_method,
            **bubble_params,
        )
        all_bubble_labels[t] = bubble_labels
        for r in results:
            r["frame"] = t
        all_results.extend(results)

        # Progress update
        yield ("progress", t + 1, n_frames)

    # --- 3. Track cells ------------------------------------------------------
    track_df = track_cells(all_masks)

    # Final results
    yield ("done", all_masks, all_bubble_labels, track_df, all_results)
