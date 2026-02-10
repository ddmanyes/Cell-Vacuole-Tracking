"""Layer management utilities for Napari viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari


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
