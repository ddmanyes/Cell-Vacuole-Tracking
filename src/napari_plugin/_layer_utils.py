"""Layer management utilities for Napari viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import napari
    import pandas as pd


# ---------------------------------------------------------------------------
# Colormaps aligned with ui_design_philosophy.md
# ---------------------------------------------------------------------------
CELL_COLORMAP = "cyan"
BUBBLE_COLORMAP = "magenta"
TRACK_COLORMAP = "yellow"


def update_or_create_labels(
    viewer: napari.Viewer,
    data: np.ndarray,
    name: str,
    *,
    opacity: float = 0.5,
) -> None:
    """Update an existing Labels layer or create a new one.

    Avoids accumulating duplicate layers when the user clicks
    'Test Current Frame' repeatedly.
    """
    existing = [layer for layer in viewer.layers if layer.name == name]
    if existing:
        existing[0].data = data
    else:
        viewer.add_labels(data, name=name, opacity=opacity)


def update_or_create_image(
    viewer: napari.Viewer,
    data: np.ndarray,
    name: str,
    *,
    colormap: str = "gray",
    opacity: float = 1.0,
) -> None:
    """Update an existing Image layer or create a new one."""
    existing = [layer for layer in viewer.layers if layer.name == name]
    if existing:
        existing[0].data = data
    else:
        viewer.add_image(data, name=name, colormap=colormap, opacity=opacity)


def update_or_create_tracks(
    viewer: napari.Viewer,
    track_df: pd.DataFrame,
    name: str,
    *,
    opacity: float = 1.0,
    colormap: str = "turbo",
) -> None:
    """Update an existing Tracks layer or create a new one.

    Converts DataFrame with [tracked_id, frame, y, x] to Napari tracks format.
    """
    if track_df.empty:
        return

    # Prepare data: (ID, T, Y, X)
    required = {"tracked_id", "frame", "y", "x"}
    if not required.issubset(track_df.columns):
        print(f"Warning: track_df missing columns {required - set(track_df.columns)}")
        return

    data = track_df[["tracked_id", "frame", "y", "x"]].to_numpy()

    # Properties for coloring (optional, can be expanded later)
    properties = {"track_id": data[:, 0]}

    existing = [layer for layer in viewer.layers if layer.name == name]
    if existing:
        existing[0].data = data
        existing[0].properties = properties
    else:
        viewer.add_tracks(
            data,
            name=name,
            properties=properties,
            opacity=opacity,
            colormap=colormap,
            tail_length=50,  # Show tail for better visualization
            tail_width=2,
        )
