"""Cell Vacuole Tracker — Napari Dock Widget.

Implements the control panel with three sections:
  1. Segmentation parameters
  2. Bubble detection parameters
  3. Execution buttons (Test Current Frame / Run Full Video)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    FloatSlider,
    FloatSpinBox,
    Label,
    PushButton,
    SpinBox,
    Table,
    create_widget,
)
from qtpy.QtWidgets import QProgressBar, QWidget, QVBoxLayout

from src.pipeline.pipeline import load_config, CONFIG

from ._analysis import compute_summary_stats, merge_track_and_bubbles
from ._layer_utils import update_or_create_labels, update_or_create_tracks
from ._worker import run_single_frame, run_full_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(section: str, key: str, default=None):
    """Retrieve a nested config value from pipeline CONFIG."""
    parts = section.split(".")
    d = CONFIG
    for p in parts:
        d = d.get(p, {})
    return d.get(key, default) if isinstance(d, dict) else default


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CellVacuoleWidget(QWidget):
    """Main dock widget for the Cell Vacuole Tracker plugin."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Reload config to get freshest values
        load_config()

        self.last_results_df = None  # Store results for export

        self._build_ui()
        self._connect_signals()

    # ---- UI Construction ---------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # === Section 1: Segmentation ========================================
        self.seg_header = Label(value="── 細胞分割 (Segmentation) ──")

        self.seg_method = ComboBox(
            label="方法",
            choices=["cellpose", "watershed"],
            value=CONFIG.get("segmentation", {}).get("method", "cellpose"),
        )

        # Cellpose params
        self.cp_diameter = SpinBox(
            label="Diameter",
            value=CONFIG.get("cellpose", {}).get("diameter", 100),
            min=10, max=500, step=10,
        )
        self.cp_cellprob = FloatSlider(
            label="Cell Prob",
            value=CONFIG.get("cellpose", {}).get("cellprob_threshold", 0.6),
            min=-6.0, max=6.0, step=0.1,
        )
        self.cp_flow = FloatSlider(
            label="Flow Thresh",
            value=CONFIG.get("cellpose", {}).get("flow_threshold", 0.4),
            min=0.0, max=3.0, step=0.1,
        )

        # Watershed params
        self.ws_sigma = FloatSpinBox(
            label="Gaussian σ",
            value=CONFIG.get("segmentation", {}).get("gaussian_sigma", 1.0),
            min=0.1, max=10.0, step=0.1,
        )
        self.ws_min_area = SpinBox(
            label="Min Cell Area",
            value=CONFIG.get("segmentation", {}).get("min_cell_area", 200),
            min=10, max=5000, step=10,
        )
        self.ws_peak_fp = SpinBox(
            label="Peak Footprint",
            value=CONFIG.get("segmentation", {}).get("peak_footprint", 7),
            min=3, max=31, step=2,
        )

        self.seg_container = Container(widgets=[
            self.seg_header,
            self.seg_method,
            self.cp_diameter,
            self.cp_cellprob,
            self.cp_flow,
            self.ws_sigma,
            self.ws_min_area,
            self.ws_peak_fp,
        ])

        # === Section 2: Bubble Detection ====================================
        self.bub_header = Label(value="── 泡泡偵測 (Bubble Detection) ──")

        self.bub_method = ComboBox(
            label="方法",
            choices=["rb_clahe", "tophat", "gradient_ws", "cellpose"],
            value=CONFIG.get("bubble", {}).get("method", "rb_clahe"),
        )

        # rb_clahe params (primary method)
        rb = CONFIG.get("bubble", {}).get("rb_clahe", {})
        self.bub_thresh = FloatSlider(
            label="Threshold",
            value=rb.get("thresh", 0.28),
            min=0.01, max=1.0, step=0.01,
        )
        self.bub_clahe = FloatSlider(
            label="CLAHE Clip",
            value=rb.get("clahe_clip", 0.5),
            min=0.01, max=2.0, step=0.01,
        )
        self.bub_min_area = SpinBox(
            label="Min Area",
            value=rb.get("min_area", 10),
            min=1, max=500, step=1,
        )
        self.bub_max_area = SpinBox(
            label="Max Area",
            value=rb.get("max_area", 0) or 0,  # None → 0 means unlimited
            min=0, max=5000, step=10,
            tooltip="0 = 不限制",
        )
        self.bub_circularity = FloatSlider(
            label="Min Circularity",
            value=rb.get("min_circularity", 0.1),
            min=0.0, max=1.0, step=0.05,
        )
        self.bub_rb_radius = SpinBox(
            label="RB Radius",
            value=rb.get("rb_radius", 50),
            min=1, max=200, step=5,
        )

        self.bub_container = Container(widgets=[
            self.bub_header,
            self.bub_method,
            self.bub_thresh,
            self.bub_clahe,
            self.bub_min_area,
            self.bub_max_area,
            self.bub_circularity,
            self.bub_rb_radius,
        ])

        # === Section 3: Execution ===========================================
        self.exec_header = Label(value="── 執行 (Execution) ──")

        self.btn_test_frame = PushButton(text="▶ Test Current Frame")
        self.btn_run_video = PushButton(text="▶▶ Run Full Video")
        self.btn_run_video.enabled = True

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.status_label = Label(value="就緒")

        self.exec_container = Container(widgets=[
            self.exec_header,
            self.btn_test_frame,
            self.btn_run_video,
            self.status_label,
        ])

        # === Section 4: Analysis ============================================
        self.ana_header = Label(value="── 分析 (Analysis) ──")

        self.table = Table(value=[])
        self.table.read_only = True
        self.table.min_height = 150

        self.btn_export = PushButton(text="💾 Export CSV")
        self.btn_export.enabled = False

        self.ana_container = Container(widgets=[
            self.ana_header,
            self.table,
            self.btn_export,
        ])

        # === Assemble ========================================================
        layout.addWidget(self.seg_container.native)
        layout.addWidget(self.bub_container.native)
        layout.addWidget(self.exec_container.native)
        layout.addWidget(self.ana_container.native)
        layout.addWidget(self.progress)
        layout.addStretch()

        # Initial visibility
        self._update_seg_visibility()
        self._update_bub_visibility()

    # ---- Signal Connections ------------------------------------------------

    def _connect_signals(self):
        self.seg_method.changed.connect(self._update_seg_visibility)
        self.bub_method.changed.connect(self._update_bub_visibility)
        self.btn_test_frame.changed.connect(self._on_test_frame)
        self.btn_run_video.changed.connect(self._on_run_video)
        self.btn_export.changed.connect(self._on_export)

    # ---- Visibility Logic --------------------------------------------------

    def _update_seg_visibility(self, *_):
        is_cellpose = self.seg_method.value == "cellpose"
        self.cp_diameter.visible = is_cellpose
        self.cp_cellprob.visible = is_cellpose
        self.cp_flow.visible = is_cellpose
        self.ws_sigma.visible = not is_cellpose
        self.ws_min_area.visible = not is_cellpose
        self.ws_peak_fp.visible = not is_cellpose

    def _update_bub_visibility(self, *_):
        is_rb = self.bub_method.value == "rb_clahe"
        self.bub_thresh.visible = is_rb
        self.bub_clahe.visible = is_rb
        self.bub_min_area.visible = True  # shared
        self.bub_max_area.visible = True
        self.bub_circularity.visible = is_rb
        self.bub_rb_radius.visible = is_rb

    # ---- Gather Parameters -------------------------------------------------

    def _get_seg_params(self) -> dict:
        """Collect segmentation parameters from the UI."""
        if self.seg_method.value == "cellpose":
            return {
                "diameter": self.cp_diameter.value,
                "cellprob_threshold": self.cp_cellprob.value,
                "flow_threshold": self.cp_flow.value,
            }
        else:
            return {
                "gaussian_sigma": self.ws_sigma.value,
                "min_cell_area": self.ws_min_area.value,
                "peak_footprint": self.ws_peak_fp.value,
            }

    def _get_bubble_params(self) -> dict:
        """Collect bubble detection parameters from the UI."""
        method = self.bub_method.value
        params: dict = {
            "min_area": self.bub_min_area.value,
            "max_area": self.bub_max_area.value if self.bub_max_area.value > 0 else None,
        }

        if method == "rb_clahe":
            params.update({
                "thresh": self.bub_thresh.value,
                "clahe_clip": self.bub_clahe.value,
                "min_circularity": self.bub_circularity.value,
                "rb_radius": self.bub_rb_radius.value,
            })

        return params

    # ---- Get Current Frame -------------------------------------------------

    def _get_current_frame(self) -> np.ndarray | None:
        """Return the current 2-D frame from the active image layer."""
        # Find the first Image layer
        for layer in self.viewer.layers:
            if layer._type_string == "image":
                data = layer.data
                if data.ndim == 2:
                    return data
                elif data.ndim >= 3:
                    # Use the current viewer step (time slider position)
                    step = self.viewer.dims.current_step
                    t = step[0] if len(step) > 0 else 0
                    t = min(t, data.shape[0] - 1)
                    return data[t]
        return None

    # ---- Execute: Test Current Frame ---------------------------------------

    def _on_test_frame(self, *_):
        frame = self._get_current_frame()
        if frame is None:
            self.status_label.value = "⚠️ 請先載入影像"
            return

        self.btn_test_frame.enabled = False
        self.status_label.value = "⏳ 處理中..."

        worker = run_single_frame(
            frame,
            seg_method=self.seg_method.value,
            seg_params=self._get_seg_params(),
            bubble_method=self.bub_method.value,
            bubble_params=self._get_bubble_params(),
        )

        worker.yielded.connect(self._on_single_frame_result)
        worker.finished.connect(self._on_test_frame_finished)
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_single_frame_result(self, result):
        cell_mask, bubble_labels = result
        update_or_create_labels(self.viewer, cell_mask, "Cell Masks", opacity=0.4)
        update_or_create_labels(self.viewer, bubble_labels, "Bubble Labels", opacity=0.6)
        self.status_label.value = (
            f"✅ 偵測完成 — "
            f"細胞: {len(np.unique(cell_mask)) - 1}, "
            f"泡泡: {len(np.unique(bubble_labels)) - 1}"
        )

    def _on_test_frame_finished(self):
        self.btn_test_frame.enabled = True

    # ---- Execute: Run Full Video -------------------------------------------

    def _on_run_video(self, *_):
        # Find image data
        img_data = None
        for layer in self.viewer.layers:
            if layer._type_string == "image":
                img_data = layer.data
                break

        if img_data is None or img_data.ndim < 3:
            self.status_label.value = "⚠️ 請先載入多幀影像 (T, Y, X)"
            return

        self.btn_run_video.enabled = False
        self.btn_test_frame.enabled = False
        self.progress.setVisible(True)
        self.progress.setMaximum(img_data.shape[0])
        self.progress.setValue(0)
        self.status_label.value = "⏳ 批次處理中..."

        worker = run_full_video(
            img_data,
            seg_method=self.seg_method.value,
            seg_params=self._get_seg_params(),
            bubble_method=self.bub_method.value,
            bubble_params=self._get_bubble_params(),
        )

        worker.yielded.connect(self._on_video_yielded)
        worker.finished.connect(self._on_video_finished)
        worker.errored.connect(self._on_worker_error)
        worker.start()

    def _on_video_yielded(self, item):
        if item[0] == "progress":
            _, current, total = item
            self.progress.setValue(current)
            self.status_label.value = f"⏳ {current}/{total} 幀..."
        elif item[0] == "done":
            _, all_masks, all_bubble_labels, track_df, all_results = item
            update_or_create_labels(self.viewer, all_masks, "Cell Masks", opacity=0.4)
            update_or_create_labels(self.viewer, all_bubble_labels, "Bubble Labels", opacity=0.6)
            update_or_create_tracks(self.viewer, track_df, "Cell Tracks")

            n_cells = len(np.unique(all_masks)) - 1
            n_bubbles = len(all_results)
            n_tracks = track_df["tracked_id"].nunique() if not track_df.empty else 0
            self.status_label.value = (
                f"✅ 完成 — {all_masks.shape[0]} 幀, "
                f"{n_cells} 細胞, {n_tracks} 軌跡, {n_bubbles} 泡泡"
            )

            # Update Analysis Table
            merged_df = merge_track_and_bubbles(track_df, all_results)
            self.last_results_df = merged_df
            
            if not merged_df.empty:
                summary_df = compute_summary_stats(merged_df)
                self.table.value = summary_df.to_dict("list")
                self.btn_export.enabled = True
            else:
                self.table.value = []
                self.btn_export.enabled = False

    def _on_video_finished(self):
        self.btn_run_video.enabled = True
        self.btn_test_frame.enabled = True
        self.progress.setVisible(False)

    def _on_export(self, *_):
        if self.last_results_df is None or self.last_results_df.empty:
            return

        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Results", "results.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                self.last_results_df.to_csv(path, index=False)
                self.status_label.value = f"💾 已匯出至 {Path(path).name}"
            except Exception as e:
                self.status_label.value = f"❌ 匯出失敗: {e}"

    # ---- Error Handling ----------------------------------------------------

    def _on_worker_error(self, exc):
        self.btn_test_frame.enabled = True
        self.btn_run_video.enabled = True
        self.progress.setVisible(False)
        self.status_label.value = f"❌ 錯誤: {exc}"
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__)
