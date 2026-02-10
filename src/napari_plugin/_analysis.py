"""Data analysis and export utilities for the plugin."""

from __future__ import annotations

from typing import Any

import pandas as pd


def merge_track_and_bubbles(
    track_df: pd.DataFrame,
    bubble_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge tracking data with bubble detection results.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with columns ['frame', 'label', 'tracked_id', 'y', 'x'].
    bubble_results : list[dict]
        List of dictionaries with keys ['frame', 'label', 'bubble_count', ...].

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both tracking and bubble metrics.
    """
    if not bubble_results:
        return track_df

    # Convert bubble results to DataFrame
    bubble_df = pd.DataFrame(bubble_results)

    # Ensure required columns exist for merging
    if track_df.empty:
        return bubble_df

    # Merge on [frame, label]
    # track_df supplies 'tracked_id', 'y', 'x'
    # bubble_df supplies 'bubble_count', 'bubble_area', 'cell_size_px'

    # Standardize types for merging
    track_df["frame"] = track_df["frame"].astype(int)
    track_df["label"] = track_df["label"].astype(int)
    bubble_df["frame"] = bubble_df["frame"].astype(int)
    bubble_df["label"] = bubble_df["label"].astype(int)

    merged = pd.merge(
        track_df,
        bubble_df,
        on=["frame", "label"],
        how="left",  # Keep all tracked cells, even if no bubble data (shouldn't happen)
    )

    # Fill NaN for bubble stats if any (e.g. tracking interpolation)
    merged["bubble_count"] = merged["bubble_count"].fillna(0).astype(int)
    merged["bubble_area"] = merged["bubble_area"].fillna(0.0)

    # Reorder columns for readability
    cols = [
        "frame",
        "tracked_id",
        "label",
        "y",
        "x",
        "cell_size_px",
        "bubble_count",
        "bubble_area",
        "bubble_density",
    ]
    # Keep only columns that exist
    final_cols = [c for c in cols if c in merged.columns]
    return merged[final_cols]


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track summary statistics.

    Returns
    -------
    pd.DataFrame
        Summary with [tracked_id, mean_bubble_count, max_bubble_area, duration]
    """
    if df.empty or "tracked_id" not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("tracked_id").agg(
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        duration=("frame", "count"),
        mean_bubble_count=("bubble_count", "mean"),
        max_bubble_count=("bubble_count", "max"),
        mean_bubble_area=("bubble_area", "mean"),
        mean_cell_size=("cell_size_px", "mean"),
    ).reset_index()

    return summary.round(2)
