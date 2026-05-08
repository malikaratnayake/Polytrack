"""
polytrack_app.py
================
PySide6 desktop application for Polytrack: Environment Pollinator Tracking.

Wired to src/main.py via QProcess.  The "Start Tracking" button:
  1. Collects the current config from all views into a dict.
  2. Writes it to a temp YAML file.
  3. Launches  python main.py --config <temp.yaml>  in a QProcess.
  4. Parses stdout (the carriage-return progress line) for live stats.

Colour scheme: EcoMotionZip dark/light palette (slate base, emerald-600 primary,
blue-500 focus).  Starts in the system-preferred theme; toggle via header button.

Run
---
    python src/polytrack_app.py
"""

from __future__ import annotations

import sys
import os
import re
import copy
import tempfile
from pathlib import Path

import yaml

from PySide6.QtCore import Qt, QProcess, QTimer, QSize, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen, QTextCursor, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QSplitter, QMessageBox,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Colour palettes — Tailwind CSS v3 tokens (EcoMotionZip)
# ═══════════════════════════════════════════════════════════════════════════════

_DARK: dict[str, str] = dict(
    bg           = "#020617",   # slate-950
    surface      = "#0F172A",   # slate-900
    panel        = "#1E293B",   # slate-800
    border       = "#334155",   # slate-700
    muted        = "#475569",   # slate-600
    placeholder  = "#64748B",   # slate-500
    text_dim     = "#94A3B8",   # slate-400
    text         = "#CBD5E1",   # slate-300
    text_bright  = "#F1F5F9",   # slate-100
    primary      = "#059669",   # emerald-600
    primary_hv   = "#047857",   # emerald-700
    primary_lt   = "#34D399",   # emerald-400
    focus        = "#3B82F6",   # blue-500
    focus_hv     = "#2563EB",   # blue-600
    warn         = "#FBBF24",   # amber-400
    error        = "#F87171",   # red-400
    stop         = "#DC2626",   # red-600
    stop_hv      = "#B91C1C",   # red-700
    pink         = "#F472B6",   # pink-400
    toggle_bg    = "#334155",   # off-state track
    log_bg       = "#0F172A",
)

_LIGHT: dict[str, str] = dict(
    bg           = "#F8FAFC",   # slate-50
    surface      = "#FFFFFF",   # white
    panel        = "#F1F5F9",   # slate-100
    border       = "#CBD5E1",   # slate-300
    muted        = "#94A3B8",   # slate-400
    placeholder  = "#94A3B8",   # slate-400
    text_dim     = "#64748B",   # slate-500
    text         = "#334155",   # slate-700
    text_bright  = "#0F172A",   # slate-900
    primary      = "#059669",   # emerald-600
    primary_hv   = "#047857",   # emerald-700
    primary_lt   = "#10B981",   # emerald-500
    focus        = "#3B82F6",   # blue-500
    focus_hv     = "#2563EB",   # blue-600
    warn         = "#D97706",   # amber-600
    error        = "#DC2626",   # red-600
    stop         = "#DC2626",   # red-600
    stop_hv      = "#B91C1C",   # red-700
    pink         = "#DB2777",   # pink-600
    toggle_bg    = "#CBD5E1",   # off-state track
    log_bg       = "#F1F5F9",
)

# Active theme — mutable dict, updated by _set_theme()
_C: dict[str, str] = dict(**_DARK)


def _set_theme(name: str) -> None:
    """Switch active theme in-place (does not rebuild UI)."""
    _C.update(_DARK if name == "dark" else _LIGHT)


def _detect_system_theme() -> str:
    """Return 'dark' or 'light' based on the OS / Qt palette."""
    try:
        # Qt 6.5+: styleHints().colorScheme() returns Qt.ColorScheme.Dark/Light
        from PySide6.QtGui import QGuiApplication
        scheme = QGuiApplication.styleHints().colorScheme()
        # Qt.ColorScheme.Dark == 2 in recent builds, but value may differ
        if hasattr(Qt, "ColorScheme"):
            if scheme == Qt.ColorScheme.Dark:
                return "dark"
            if scheme == Qt.ColorScheme.Light:
                return "light"
    except Exception:
        pass
    # Fallback: check window background lightness
    pal = QApplication.palette()
    return "dark" if pal.window().color().lightness() < 128 else "light"


# ═══════════════════════════════════════════════════════════════════════════════
# QSS generator
# ═══════════════════════════════════════════════════════════════════════════════

def build_qss(c: dict[str, str]) -> str:
    return f"""
/* ═══════════════════════════════════════════════════
   Polytrack  —  dark/light stylesheet
   ═══════════════════════════════════════════════════ */

QMainWindow, QDialog {{ background-color: {c['bg']}; }}
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "Segoe UI", "SF Pro Text", "Ubuntu", Arial, sans-serif;
    font-size: 13px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {c['bg']}; border: none; }}

/* Transparent containers */
QLabel, QCheckBox, QFrame, QAbstractScrollArea {{ background-color: transparent; }}
QGroupBox QWidget     {{ background-color: transparent; }}
QGroupBox QLineEdit,
QGroupBox QSpinBox,
QGroupBox QDoubleSpinBox,
QGroupBox QComboBox   {{ background-color: {c['panel']}; }}

/* Header */
QWidget#header {{ background-color: {c['surface']}; border-bottom: 1px solid {c['border']}; }}

/* Sidebar */
QWidget#sidebar {{ background-color: {c['surface']}; }}

/* Group boxes */
QGroupBox {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    margin-top: 16px;
    padding: 14px 12px 12px 12px;
    font-size: 11px;
    font-weight: 700;
    color: {c['text_dim']};
    letter-spacing: 0.8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 2px 8px;
    background-color: transparent;
    color: {c['text_dim']};
}}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text_bright']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    selection-background-color: {c['focus']};
    selection-color: #FFFFFF;
    min-height: 20px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1.5px solid {c['focus']};
}}
QWidget:disabled {{ color: {c['muted']}; }}
QLabel:disabled  {{ color: {c['muted']}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['muted']};
    background-color: {c['bg']};
    border-color: {c['border']};
}}
QLineEdit::placeholder-text {{ color: {c['placeholder']}; font-style: italic; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 20px; background: {c['border']}; border-radius: 3px; margin: 1px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {c['muted']};
}}
QComboBox::drop-down {{ border: none; width: 28px; background: {c['border']}; border-radius: 0 5px 5px 0; }}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text_bright']};
    border: 1px solid {c['border']};
    selection-background-color: {c['focus']};
    selection-color: #FFFFFF;
    outline: none;
    padding: 3px;
}}

/* Checkboxes */
QCheckBox {{ color: {c['text']}; spacing: 8px; font-size: 13px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px; border-radius: 4px;
    border: 1.5px solid {c['border']}; background: {c['panel']};
}}
QCheckBox::indicator:hover   {{ border-color: {c['focus']}; }}
QCheckBox::indicator:checked {{ background: {c['focus']}; border-color: {c['focus']}; }}

/* Buttons */
QPushButton {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover   {{ background-color: {c['border']}; color: {c['text_bright']}; }}
QPushButton:pressed {{ background-color: {c['muted']};  color: {c['text_bright']}; }}
QPushButton:disabled {{ background-color: {c['panel']}; color: {c['muted']}; border-color: {c['border']}; }}

/* Browse buttons */
QPushButton#browse_btn {{
    background-color: {c['border']};
    color: {c['text']};
    border: 1px solid {c['muted']};
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
}}
QPushButton#browse_btn:hover {{ background-color: {c['muted']}; color: {c['text_bright']}; }}

/* Progress bars */
QProgressBar {{
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    color: transparent;
    min-height: 8px;
    max-height: 14px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['focus']}, stop:1 {c['primary']});
    border-radius: 5px;
}}

/* List widget (sidebar nav) */
QListWidget {{ background: transparent; border: none; outline: none; padding: 4px 0; }}
QListWidget::item {{
    color: {c['text_dim']};
    padding: 10px 16px;
    border-radius: 8px;
    margin: 2px 8px;
    font-weight: 500;
}}
QListWidget::item:hover    {{ background: rgba(59,130,246,0.12); color: {c['text']}; }}
QListWidget::item:selected {{ background: rgba(59,130,246,0.18); color: {c['primary_lt']}; font-weight: 700; }}

/* Scroll bars */
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; border: none; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; border: none; }}
QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* H-line separators */
QFrame[frameShape="4"] {{ background: {c['border']}; border: none; max-height: 1px; min-height: 1px; }}

/* Tooltips */
QToolTip {{ background: {c['panel']}; color: {c['text_bright']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 6px 10px; }}
"""


# ── Default config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "directories": {
        "source": "/Volumes/MalikaSSD/WhiteCloud-Oct2025/Trial_1/AGC2/AGC2_251004/AGC2_251004-S/",
        "output": "/Users/mrat0010/Documents/Costa_Data/Output1/",
        "_input_mode": "Directory",
    },
    "source": {
        "compressed_video": False,
        "compression_info": "",
        "skip_frames": False,
        "device": "auto",
    },
    "output": {
        "resolution": [1920, 1080],
        "codec": "mp4v",
        "show": True,
        "save": True,
        "compressed_time_as_filename": False,
        "save_insect_snapshots": False,
        "log_level": "INFO",
    },
    "insect_tracking": {
        "labels": ["bee"],
        "classes": [0],
        "prediction_method": ["ConstantVelocity"],
        "assignment_method": ["HungarianMethod"],
        "detectors": ["dl_detection", "fgbg_detection"],
        "detection_interval": 300,
        "min_blob_area": 5,
        "max_blob_area": 8000,
        "insect_boundary_extension": 2.5,
        "iou_threshold": 0,
        "max_occlusions": 60,
        "max_occlusions_on_flower": 150,
        "missing_jump_scale": 1.5,
        "new_track_distance_thresh": 5,
        "jump_distance": [50, 50],
        "min_track_length": 10,
        "record_yolo_bbox": True,
        "compressed_video_time_jump": 100,
        "edge_analysis": {
            "edge_pixels": 40,
            "max_edge_occlusions": 32,
            "continious_analysis": False,
            "compressed_video_time_jump": 100,
        },
        "detector_properties": {
            "dl_detection": {
                "model": "/Users/mrat0010/Documents/Costa_Data/Detection_Models/insects.pt",
                "detection_confidence": [0.2],
                "detection_confidence_floor": [0.001],
                "use_fp16": True,
                "iou_threshold": 0,
                "image_size": [1024, 1024],
            },
            "secondary_verification": {
                "model": "/Users/mrat0010/Documents/Costa_Aug_2025/Models/11.pt",
                "detection_confidence": [0.5],
                "black_pixel_threshold": 0.5,
                "image_size": [640, 640],
            },
            "fgbg_detection": {
                "model": ["MOG2"],
                "downscale_factor": 1,
                "dilate_kernel_size": 4,
                "movement_threshold": 80,
                "warmup_frames": 5,
                "show": False,
                "clean_detections": False,
                "clean_distance_thresh": 3.0,
                "clean_area_ratio_thresh": 0.25,
                "max_fgbg_candidates": 500,
            },
        },
        "spatial_filtering": {
            "use_include_zone": False,
            "include_zone_coord": [500, 200, 3500, 2500],
            "use_exclude_zone": False,
            "exclude_zone_coord": [775, 300, 950, 750],
        },
    },
    "flower_tracking": {
        "track": True,
        "output_shape": "Circle",
        "show_border_extension": True,
        "mark_insects_on_flower": True,
        "labels": ["flower"],
        "classes": [0],
        "detection_interval": 4000,
        "border_extension": 2.5,
        "prediction_method": "ConstantVelocity",
        "detector_properties": {
            "model": "/Users/mrat0010/Documents/Costa_Data/Detection_Models/flowers.pt",
            "detection_confidence": 0.1,
            "use_fp16": True,
            "image_size": [1024, 1024],
            "iou_threshold": 0.95,
        },
    },
}


def _deep_set(d: dict, path: list, value) -> None:
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def _config_for_yaml(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.get("directories", {}).pop("_input_mode", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Shared widget helpers  (all read _C at call-time, not at import-time)
# ═══════════════════════════════════════════════════════════════════════════════

class ToggleSwitch(QWidget):
    """Sliding pill toggle — primary colour when on, muted when off."""
    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(44, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background: transparent;")

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, v: bool) -> None:
        self._checked = bool(v)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor(_C['primary'] if self._checked else _C['toggle_bg'])
        p.setBrush(QBrush(track))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 44, 24, 12, 12)
        handle_x = 22 if self._checked else 2
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawEllipse(handle_x, 2, 20, 20)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)


def _lbl(text: str, size: int = 13, bold: bool = False,
         color: str | None = None) -> QLabel:
    w = QLabel(text)
    f = w.font(); f.setPointSize(size); f.setBold(bold); w.setFont(f)
    w.setStyleSheet(f"color:{color or _C['text']}; background:transparent;")
    return w


def _muted(text: str) -> QLabel:
    w = _lbl(text, size=11, color=_C['text_dim'])
    w.setWordWrap(True)
    return w


def _combo(options: list[str], current: str = "") -> QComboBox:
    cb = QComboBox()
    cb.addItems(options)
    idx = cb.findText(current)
    if idx >= 0:
        cb.setCurrentIndex(idx)
    return cb


def _spin(value: int, lo: int = 0, hi: int = 999999, suffix: str = "") -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(value)
    if suffix:
        sp.setSuffix(suffix)
    return sp


def _dspin(value: float, lo: float = 0.0, hi: float = 100.0,
           step: float = 0.1, dec: int = 2) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(dec)
    sp.setValue(value)
    return sp


def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{_C['border']}; background:{_C['border']};")
    f.setFixedHeight(1)
    return f


def _scrolled(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    return sa


def _input_group(label: str, widget: QWidget, tooltip: str = "") -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(4)
    hdr = QHBoxLayout()
    hdr.addWidget(_lbl(label, size=12, bold=True, color=_C['text']))
    if tooltip:
        hdr.addStretch()
        hdr.addWidget(_lbl(tooltip, size=10, color=_C['placeholder']))
    lay.addLayout(hdr)
    lay.addWidget(widget)
    return lay


def _toggle_row(label: str, description: str, toggle: ToggleSwitch) -> QHBoxLayout:
    row = QHBoxLayout()
    col = QVBoxLayout(); col.setSpacing(2)
    col.addWidget(_lbl(label, size=13, bold=True, color=_C['text']))
    if description:
        col.addWidget(_muted(description))
    row.addLayout(col, 1)
    row.addWidget(toggle)
    return row


def _no_border(widget):
    """Strip the border from an input widget (QLineEdit / QSpinBox / QComboBox etc.)."""
    cls = type(widget).__name__
    widget.setStyleSheet(f"{cls} {{ border: none; }}")
    return widget


def _browse_btn(text: str = "Browse", width: int = 82) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("browse_btn")
    btn.setFixedWidth(width)
    return btn


def _model_path_row(line_edit: QLineEdit, parent_widget=None) -> QWidget:
    """Return a QWidget containing the line edit + a Browse button for a model file."""
    row = QWidget()
    row.setStyleSheet("background:transparent;")
    lay = QHBoxLayout(row); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
    lay.addWidget(line_edit, 1)
    btn = _browse_btn("Browse", 72)
    def _pick():
        path, _ = QFileDialog.getOpenFileName(
            parent_widget, "Select Model File", line_edit.text(),
            "Model Files (*.pt *.pth *.weights *.onnx);;All Files (*)")
        if path:
            line_edit.setText(path)
    btn.clicked.connect(_pick)
    lay.addWidget(btn)
    return row


def _collapsible_section(title: str = "Advanced Settings"):
    """Return (button, body_widget, body_layout) for an inline collapsible section."""
    btn = QPushButton(f"▶  {title}")
    btn.setCheckable(True); btn.setChecked(False)
    btn.setStyleSheet(
        f"QPushButton {{ background:transparent; color:{_C['text_dim']}; border:none;"
        f"padding:4px 0; font-size:12px; font-weight:600; text-align:left; }}"
        f"QPushButton:hover {{ color:{_C['text']}; }}")
    body = QWidget(); body.setVisible(False)
    lay = QVBoxLayout(body); lay.setContentsMargins(0, 4, 0, 0); lay.setSpacing(8)
    def _toggle(on: bool):
        btn.setText(("▼" if on else "▶") + f"  {title}")
        body.setVisible(on)
    btn.toggled.connect(_toggle)
    return btn, body, lay


class CardFrame(QFrame):
    """Surface-coloured card with rounded border."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        # Single f-string — both {{ }} pairs are in the same expression
        self.setStyleSheet(f"#CardFrame {{ background:{_C['surface']}; border:1px solid {_C['border']}; border-radius:10px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)


class DetectorCard(QWidget):
    """Collapsible card with an enable toggle in its header."""

    def __init__(self, title: str, enabled: bool, parent=None):
        super().__init__(parent)
        self._enabled = enabled
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._border = QFrame()
        self._border.setStyleSheet(f"QFrame {{ background:{_C['surface']}; border:1px solid {_C['border']}; border-radius:12px; }}")
        outer = QVBoxLayout(self._border)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QWidget()
        self._header.setStyleSheet(f"background:{_C['panel']}; border-radius:10px 10px 0 0;")
        hdr_lay = QHBoxLayout(self._header)
        hdr_lay.setContentsMargins(16, 12, 16, 12)
        self._title_lbl = _lbl(title, size=13, bold=True,
                                color=_C['text_bright'] if enabled else _C['muted'])
        hdr_lay.addWidget(self._title_lbl, 1)
        self._toggle = ToggleSwitch(enabled)
        self._toggle.toggled.connect(self._on_toggle)
        hdr_lay.addWidget(self._toggle)
        outer.addWidget(self._header)

        self._body = QWidget()
        self._body.setVisible(enabled)
        self._body_layout = QGridLayout(self._body)
        self._body_layout.setContentsMargins(16, 12, 16, 16)
        self._body_layout.setSpacing(10)
        outer.addWidget(self._body)

        root.addWidget(self._border)

    def body_layout(self) -> QGridLayout:
        return self._body_layout

    def _on_toggle(self, checked: bool):
        self._enabled = checked
        self._title_lbl.setStyleSheet(f"color:{_C['text_bright'] if checked else _C['muted']}; background:transparent;")
        self._body.setVisible(checked)

    @property
    def is_enabled(self) -> bool:
        return self._toggle.isChecked()

    @property
    def enabled_signal(self):
        return self._toggle.toggled


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

class PathsView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        self._c = config
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(20)
        root.addWidget(_lbl("Paths & Directories", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted("Set the locations for your source videos and where output will be saved."))
        root.addWidget(_hsep())

        inp = CardFrame()
        inp.layout().addWidget(_lbl("Video Source", size=13, bold=True, color=_C['text_bright']))
        inp.layout().addWidget(_hsep())

        self._src_le = QLineEdit(config["directories"]["source"])
        self._src_le.setPlaceholderText("Select a video file, multiple files, or a folder…")
        self._src_le.textChanged.connect(lambda v: update_fn(["directories", "source"], v))

        btn_files = _browse_btn("File(s)", 68)
        btn_files.setToolTip("Select one or more video files")
        btn_files.clicked.connect(self._browse_video_files)
        btn_folder = _browse_btn("Folder", 68)
        btn_folder.setToolTip("Select a folder — all videos inside will be processed")
        btn_folder.clicked.connect(self._browse_video_folder)

        inp.layout().addWidget(_lbl("Source Path", size=12, bold=True, color=_C['text']))
        src_row = QHBoxLayout(); src_row.setSpacing(8)
        src_row.addWidget(self._src_le, 1)
        src_row.addWidget(btn_files)
        src_row.addWidget(btn_folder)
        inp.layout().addLayout(src_row)

        # ── EcoMotionZip Compressed — directly below source path ──────────────
        inp.layout().addWidget(_hsep())
        comp_toggle = ToggleSwitch(config["source"]["compressed_video"])
        comp_toggle.toggled.connect(lambda v: update_fn(["source", "compressed_video"], v))
        inp.layout().addLayout(_toggle_row("EcoMotionZip Compressed",
            "Enable if the source video was compressed using EcoMotionZip.", comp_toggle))

        comp_info_wrap = QWidget()
        ci_lay = QVBoxLayout(comp_info_wrap); ci_lay.setContentsMargins(16, 4, 0, 0); ci_lay.setSpacing(4)
        comp_info_le = QLineEdit(config["source"]["compression_info"])
        comp_info_le.setPlaceholderText("Path to compression info CSV…")
        comp_info_le.textChanged.connect(lambda v: update_fn(["source", "compression_info"], v))
        btn_comp = _browse_btn("Browse", 82)
        btn_comp.clicked.connect(lambda: comp_info_le.setText(
            QFileDialog.getOpenFileName(self, "Select Compression Info CSV",
                                        str(Path.home()), "CSV files (*.csv);;All Files (*)")[0]
            or comp_info_le.text()))
        comp_info_le_row = QHBoxLayout(); comp_info_le_row.setSpacing(8)
        comp_info_le_row.addWidget(comp_info_le, 1); comp_info_le_row.addWidget(btn_comp)
        ci_lay.addWidget(_lbl("Compression Info File", size=12, bold=True, color=_C['text']))
        ci_lay.addLayout(comp_info_le_row)
        comp_info_wrap.setVisible(config["source"]["compressed_video"])
        comp_toggle.toggled.connect(comp_info_wrap.setVisible)
        inp.layout().addWidget(comp_info_wrap)

        # ── Advanced Settings (collapsible) — contains Skip Frames ───────────
        inp.layout().addWidget(_hsep())
        adv_btn = QPushButton("▶  Advanced Settings")
        adv_btn.setCheckable(True)
        adv_btn.setChecked(False)
        adv_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{_C['text_dim']}; border:none;"
            f"padding:4px 0; font-size:12px; font-weight:600; text-align:left; }}"
            f"QPushButton:hover {{ color:{_C['text']}; }}")
        adv_body = QWidget(); adv_body.setVisible(False)
        adv_body_lay = QVBoxLayout(adv_body)
        adv_body_lay.setContentsMargins(0, 6, 0, 0); adv_body_lay.setSpacing(8)

        skip_toggle = ToggleSwitch(config["source"]["skip_frames"])
        skip_toggle.toggled.connect(lambda v: update_fn(["source", "skip_frames"], v))
        adv_body_lay.addLayout(_toggle_row("Skip Frames",
            "Process only odd frames – useful for de-interlaced video.", skip_toggle))

        def _toggle_adv(on: bool):
            adv_btn.setText(("▼" if on else "▶") + "  Advanced Settings")
            adv_body.setVisible(on)
        adv_btn.toggled.connect(_toggle_adv)

        inp.layout().addWidget(adv_btn)
        inp.layout().addWidget(adv_body)

        root.addWidget(inp)

        out = CardFrame()
        out.layout().addWidget(_lbl("Output Directory", size=13, bold=True, color=_C['text_bright']))
        out.layout().addWidget(_hsep())

        self._out_le = QLineEdit(config["directories"]["output"])
        self._out_le.setPlaceholderText("Select an output folder…")
        self._out_le.textChanged.connect(lambda v: update_fn(["directories", "output"], v))
        btn_out = _browse_btn("Browse", 82)
        btn_out.clicked.connect(self._browse_output)

        out.layout().addWidget(_lbl("Output Path", size=12, bold=True, color=_C['text']))
        out_row = QHBoxLayout(); out_row.setSpacing(8)
        out_row.addWidget(self._out_le, 1)
        out_row.addWidget(btn_out)
        out.layout().addLayout(out_row)
        root.addWidget(out)
        root.addStretch()

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))

    def _browse_video_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV);;All Files (*)")
        if paths:
            self._src_le.setText(paths[0] if len(paths) == 1 else os.pathsep.join(paths))

    def _browse_video_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Video Folder", str(Path.home()))
        if path:
            self._src_le.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(Path.home()))
        if path:
            self._out_le.setText(path)


class IOConfigView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(20)
        root.addWidget(_lbl("Input & Output Settings", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted("Configure video processing parameters, resolution, and saving options."))
        root.addWidget(_hsep())

        show_card = CardFrame()
        show_toggle = ToggleSwitch(config["output"]["show"])
        show_toggle.toggled.connect(lambda v: update_fn(["output", "show"], v))
        show_card.layout().addLayout(_toggle_row("Show Video During Processing",
            "Displays the live annotated feed (disabling this speeds up processing).", show_toggle))
        root.addWidget(show_card)

        save_card = CardFrame()
        save_toggle = ToggleSwitch(config["output"]["save"])
        save_toggle.toggled.connect(lambda v: update_fn(["output", "save"], v))
        save_card.layout().addLayout(_toggle_row("Save Output Video",
            "Save the annotated tracking video to the output directory.", save_toggle))

        save_extra = QWidget(); save_extra.setVisible(config["output"]["save"])
        save_toggle.toggled.connect(save_extra.setVisible)
        se_lay = QVBoxLayout(save_extra); se_lay.setContentsMargins(0, 8, 0, 0); se_lay.setSpacing(12)
        se_lay.addWidget(_hsep())

        # Standard resolution presets — (label, width, height)
        _RES_PRESETS = [
            ("Custom",        0,    0),
            ("720p  — 1280 × 720",    1280,  720),
            ("1080p — 1920 × 1080",   1920, 1080),
            ("1440p — 2560 × 1440",   2560, 1440),
            ("4K    — 3840 × 2160",   3840, 2160),
            ("480p  — 854 × 480",      854,  480),
            ("360p  — 640 × 360",      640,  360),
            ("2K    — 2048 × 1080",   2048, 1080),
            ("Cinema 4K — 4096 × 2160", 4096, 2160),
        ]

        w_spin = _spin(config["output"]["resolution"][0], 320, 7680)
        h_spin = _spin(config["output"]["resolution"][1], 240, 4320)

        def _find_preset_label(w: int, h: int) -> str:
            for lbl, pw, ph in _RES_PRESETS:
                if pw == w and ph == h:
                    return lbl
            return "Custom"

        preset_combo = _combo([p[0] for p in _RES_PRESETS],
                               _find_preset_label(*config["output"]["resolution"]))

        manual_row = QWidget()
        mr_lay = QHBoxLayout(manual_row); mr_lay.setContentsMargins(0, 0, 0, 0); mr_lay.setSpacing(8)
        rc_w = QVBoxLayout(); rc_w.setSpacing(4)
        rc_w.addWidget(_lbl("Width (px)", size=12, bold=True, color=_C['text'])); rc_w.addWidget(w_spin)
        rc_h = QVBoxLayout(); rc_h.setSpacing(4)
        rc_h.addWidget(_lbl("Height (px)", size=12, bold=True, color=_C['text'])); rc_h.addWidget(h_spin)
        mr_lay.addLayout(rc_w, 1); mr_lay.addLayout(rc_h, 1)
        manual_row.setVisible(preset_combo.currentText() == "Custom")

        def _on_preset_changed(label: str):
            for lbl, pw, ph in _RES_PRESETS:
                if lbl == label and pw != 0:
                    w_spin.setValue(pw); h_spin.setValue(ph)
                    break
            manual_row.setVisible(label == "Custom")

        preset_combo.currentTextChanged.connect(_on_preset_changed)

        w_spin.valueChanged.connect(lambda v: update_fn(["output", "resolution"],
            [v, config["output"]["resolution"][1]]))
        h_spin.valueChanged.connect(lambda v: update_fn(["output", "resolution"],
            [config["output"]["resolution"][0], v]))

        se_lay.addLayout(_input_group("Output Resolution", preset_combo))
        se_lay.addWidget(manual_row)

        codec_combo = _combo(["mp4v", "avc1", "h264"], config["output"]["codec"])
        codec_combo.currentTextChanged.connect(lambda v: update_fn(["output", "codec"], v))
        se_lay.addLayout(_input_group("Video Codec", codec_combo))

        yolo_bbox_toggle = ToggleSwitch(config["insect_tracking"]["record_yolo_bbox"])
        yolo_bbox_toggle.toggled.connect(lambda v: update_fn(["insect_tracking", "record_yolo_bbox"], v))
        se_lay.addWidget(_hsep())
        se_lay.addLayout(_toggle_row("Record Detection Bounding Boxes to CSV",
            "Save YOLO bounding box coordinates for each detection to a CSV file.", yolo_bbox_toggle))

        io_adv_btn, io_adv_body, io_adv_lay = _collapsible_section()
        ctf_toggle = ToggleSwitch(config["output"]["compressed_time_as_filename"])
        ctf_toggle.toggled.connect(lambda v: update_fn(["output", "compressed_time_as_filename"], v))
        io_adv_lay.addLayout(_toggle_row("Use Compressed Video Timestamps as Filenames",
            "Name output files using timestamps from EcoMotionZip metadata.", ctf_toggle))
        se_lay.addWidget(_hsep())
        se_lay.addWidget(io_adv_btn)
        se_lay.addWidget(io_adv_body)
        save_card.layout().addWidget(save_extra)
        root.addWidget(save_card)

        snap_card = CardFrame()
        snap_toggle = ToggleSwitch(config["output"]["save_insect_snapshots"])
        snap_toggle.toggled.connect(lambda v: update_fn(["output", "save_insect_snapshots"], v))
        snap_card.layout().addLayout(_toggle_row("Save Per-Insect Snapshot Images",
            "Crop and save individual insect images for manual verification.", snap_toggle))
        root.addWidget(snap_card)

        log_card = CardFrame()
        log_card.layout().addWidget(_lbl("System Logging", size=13, bold=True, color=_C['text_bright']))
        log_card.layout().addWidget(_hsep())
        log_combo = _combo(["DEBUG", "INFO", "WARNING", "ERROR"], config["output"]["log_level"])
        log_combo.currentTextChanged.connect(lambda v: update_fn(["output", "log_level"], v))
        log_card.layout().addLayout(_input_group("Console & File Log Level", log_combo))
        root.addWidget(log_card)

        root.addStretch()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))


class InsectGeneralView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        c = config["insect_tracking"]
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(20)
        root.addWidget(_lbl("Insect Tracking: General Settings", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted("Core logic for identifying, bounding, and filtering insect tracks."))
        root.addWidget(_hsep())

        # ── Tracking Configuration ────────────────────────────────────────────
        trk_cfg = CardFrame()
        trk_cfg.layout().addWidget(_lbl("Tracking Configuration", 13, True, _C['text_bright']))
        trk_cfg.layout().addWidget(_hsep())

        labels_le = QLineEdit(", ".join(c["labels"]))
        labels_le.textChanged.connect(lambda v: update_fn(["insect_tracking", "labels"],
            [x.strip() for x in v.split(",") if x.strip()]))
        trk_cfg.layout().addLayout(_input_group("Tracking Labels", labels_le, "comma-separated"))

        pred_cb = _combo(["ConstantVelocity", "KalmanFilter", "ExtendedKalmanFilter"],
                         c["prediction_method"][0])
        pred_cb.currentTextChanged.connect(
            lambda v: update_fn(["insect_tracking", "prediction_method"], [v]))
        trk_cfg.layout().addLayout(_input_group("Prediction Method", pred_cb))

        assign_cb = _combo(["HungarianMethod", "ABP"], c["assignment_method"][0])
        assign_cb.currentTextChanged.connect(
            lambda v: update_fn(["insect_tracking", "assignment_method"], [v]))
        trk_cfg.layout().addLayout(_input_group("Assignment Method", assign_cb))

        trk_cfg.layout().addWidget(_hsep())

        blob_row = QHBoxLayout(); blob_row.setSpacing(8)
        min_blob = _spin(c["min_blob_area"], 1, 999999)
        max_blob = _spin(c["max_blob_area"], 1, 999999)
        min_blob.valueChanged.connect(lambda v: update_fn(["insect_tracking", "min_blob_area"], v))
        max_blob.valueChanged.connect(lambda v: update_fn(["insect_tracking", "max_blob_area"], v))
        bc_min = QVBoxLayout(); bc_min.setSpacing(4)
        bc_min.addWidget(_lbl("Min Blob Area", 12, True, _C['text'])); bc_min.addWidget(min_blob)
        bc_max = QVBoxLayout(); bc_max.setSpacing(4)
        bc_max.addWidget(_lbl("Max Blob Area", 12, True, _C['text'])); bc_max.addWidget(max_blob)
        blob_row.addLayout(bc_min, 1); blob_row.addLayout(bc_max, 1)
        trk_cfg.layout().addLayout(blob_row)

        root.addWidget(trk_cfg)

        # ── Occlusion Management ──────────────────────────────────────────────
        occ_card = CardFrame()
        occ_card.layout().addWidget(_lbl("Occlusion Management", 13, True, _C['text_bright']))
        occ_card.layout().addWidget(_hsep())

        occ_row = QHBoxLayout(); occ_row.setSpacing(8)
        max_occ = _spin(c["max_occlusions"], 0, 9999)
        max_occ.valueChanged.connect(lambda v: update_fn(["insect_tracking", "max_occlusions"], v))
        max_occ_fl = _spin(c["max_occlusions_on_flower"], 0, 9999)
        max_occ_fl.valueChanged.connect(
            lambda v: update_fn(["insect_tracking", "max_occlusions_on_flower"], v))
        oc_a = QVBoxLayout(); oc_a.setSpacing(4)
        oc_a.addWidget(_lbl("Max Occlusions", 12, True, _C['text'])); oc_a.addWidget(max_occ)
        oc_b = QVBoxLayout(); oc_b.setSpacing(4)
        oc_b.addWidget(_lbl("Max Occlusions (Flower)", 12, True, _C['text'])); oc_b.addWidget(max_occ_fl)
        occ_row.addLayout(oc_a, 1); occ_row.addLayout(oc_b, 1)
        occ_card.layout().addLayout(occ_row)
        root.addWidget(occ_card)

        # ── Track Parameters ──────────────────────────────────────────────────
        trk_card = CardFrame()
        trk_card.layout().addWidget(_lbl("Track Parameters", 13, True, _C['text_bright']))
        trk_card.layout().addWidget(_hsep())

        tp_row1 = QHBoxLayout(); tp_row1.setSpacing(8)
        det_int = _spin(c["detection_interval"], 1, 99999)
        det_int.valueChanged.connect(lambda v: update_fn(["insect_tracking", "detection_interval"], v))
        min_trk = _spin(c["min_track_length"], 1, 9999)
        min_trk.valueChanged.connect(lambda v: update_fn(["insect_tracking", "min_track_length"], v))
        tp_a = QVBoxLayout(); tp_a.setSpacing(4)
        tp_a.addWidget(_lbl("Detection Interval", 12, True, _C['text'])); tp_a.addWidget(det_int)
        tp_b = QVBoxLayout(); tp_b.setSpacing(4)
        tp_b.addWidget(_lbl("Min Track Length", 12, True, _C['text'])); tp_b.addWidget(min_trk)
        tp_row1.addLayout(tp_a, 1); tp_row1.addLayout(tp_b, 1)
        trk_card.layout().addLayout(tp_row1)

        trk_card.layout().addWidget(_hsep())

        jump_row = QHBoxLayout(); jump_row.setSpacing(8)
        jh = _spin(c["jump_distance"][0], 1, 9999)
        jv = _spin(c["jump_distance"][1], 1, 9999)
        jh.valueChanged.connect(lambda v: update_fn(["insect_tracking", "jump_distance"],
            [v, config["insect_tracking"]["jump_distance"][1]]))
        jv.valueChanged.connect(lambda v: update_fn(["insect_tracking", "jump_distance"],
            [config["insect_tracking"]["jump_distance"][0], v]))
        jc_h = QVBoxLayout(); jc_h.setSpacing(4)
        jc_h.addWidget(_lbl("Max Jump — DL Detector (px)", 12, True, _C['text'])); jc_h.addWidget(jh)
        jc_v = QVBoxLayout(); jc_v.setSpacing(4)
        jc_v.addWidget(_lbl("Max Jump — FGBG Detector (px)", 12, True, _C['text'])); jc_v.addWidget(jv)
        jump_row.addLayout(jc_h, 1); jump_row.addLayout(jc_v, 1)
        trk_card.layout().addLayout(jump_row)

        # ── Advanced Settings (collapsible) ───────────────────────────────────
        trk_card.layout().addWidget(_hsep())
        ig_adv_btn, ig_adv_body, ig_adv_lay = _collapsible_section()

        cls_le = QLineEdit(", ".join(map(str, c["classes"])))
        cls_le.textChanged.connect(lambda v: update_fn(["insect_tracking", "classes"],
            [int(x) for x in v.split(",") if x.strip().lstrip("-").isdigit()]))
        ig_adv_lay.addLayout(_input_group("YOLO Classes (comma-separated)", cls_le,
                                          "e.g. 0, 1  — filters detections to these class IDs"))

        ig_row1 = QHBoxLayout(); ig_row1.setSpacing(8)
        bnd_ext = _dspin(c["insect_boundary_extension"], 0.1, 20.0)
        bnd_ext.valueChanged.connect(lambda v: update_fn(["insect_tracking", "insect_boundary_extension"], v))
        iou_thr = _dspin(c["iou_threshold"], 0.0, 1.0)
        iou_thr.valueChanged.connect(lambda v: update_fn(["insect_tracking", "iou_threshold"], v))
        ig_a = QVBoxLayout(); ig_a.setSpacing(4)
        ig_a.addWidget(_lbl("Boundary Extension", 12, True, _C['text'])); ig_a.addWidget(bnd_ext)
        ig_b = QVBoxLayout(); ig_b.setSpacing(4)
        ig_b.addWidget(_lbl("IoU Threshold", 12, True, _C['text'])); ig_b.addWidget(iou_thr)
        ig_row1.addLayout(ig_a, 1); ig_row1.addLayout(ig_b, 1)
        ig_adv_lay.addLayout(ig_row1)

        ig_row2 = QHBoxLayout(); ig_row2.setSpacing(8)
        mjs = _dspin(c["missing_jump_scale"], 0.1, 10.0)
        mjs.valueChanged.connect(lambda v: update_fn(["insect_tracking", "missing_jump_scale"], v))
        njdt = _spin(c["new_track_distance_thresh"], 1, 9999)
        njdt.valueChanged.connect(lambda v: update_fn(["insect_tracking", "new_track_distance_thresh"], v))
        ig_c = QVBoxLayout(); ig_c.setSpacing(4)
        ig_c.addWidget(_lbl("Missing Jump Scale", 12, True, _C['text'])); ig_c.addWidget(mjs)
        ig_d = QVBoxLayout(); ig_d.setSpacing(4)
        ig_d.addWidget(_lbl("New Track Distance Threshold", 12, True, _C['text'])); ig_d.addWidget(njdt)
        ig_row2.addLayout(ig_c, 1); ig_row2.addLayout(ig_d, 1)
        ig_adv_lay.addLayout(ig_row2)

        cvtj = _spin(c["compressed_video_time_jump"], 1, 9999)
        cvtj.valueChanged.connect(lambda v: update_fn(["insect_tracking", "compressed_video_time_jump"], v))
        ig_adv_lay.addLayout(_input_group("Compressed Video Time Jump (frames)", cvtj,
                                          "Frame gap used when skipping in compressed video"))

        trk_card.layout().addWidget(ig_adv_btn)
        trk_card.layout().addWidget(ig_adv_body)
        root.addWidget(trk_card)

        root.addStretch()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))


class InsectDetectorsView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        c  = config["insect_tracking"]
        dp = c["detector_properties"]
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(16)
        root.addWidget(_lbl("Insect Detectors", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted(
            "Configure the primary YOLO model, optional secondary verifier, "
            "and the foreground-background (FGBG) blob detector."))
        root.addWidget(_hsep())

        dl_card = DetectorCard("Primary DL Detection (YOLO)", "dl_detection" in c["detectors"])
        dl_card.enabled_signal.connect(
            lambda on: self._toggle_detector(config, update_fn, "dl_detection", on))
        g = dl_card.body_layout()
        dl_model_le = _no_border(QLineEdit(dp["dl_detection"]["model"]))
        dl_model_le.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "model"], v))
        g.addLayout(_input_group("Model Path", _model_path_row(dl_model_le, self)), 0, 0, 1, 2)
        dl_conf = _no_border(_dspin(dp["dl_detection"]["detection_confidence"][0], 0.0, 1.0))
        dl_conf.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "detection_confidence"], [v]))
        g.addLayout(_input_group("Detection Confidence", dl_conf), 1, 0)
        dl_sz = _no_border(QLineEdit(", ".join(map(str, dp["dl_detection"]["image_size"]))))
        dl_sz.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "image_size"],
            [int(x) for x in v.split(",") if x.strip().isdigit()]))
        g.addLayout(_input_group("Image Size (W, H)", dl_sz), 1, 1)
        dl_fp16 = QCheckBox("Use FP16 (CUDA only)")
        dl_fp16.setChecked(dp["dl_detection"]["use_fp16"])
        dl_fp16.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "use_fp16"], v))
        g.addWidget(dl_fp16, 2, 0, 1, 2)

        dl_adv_btn, dl_adv_body, dl_adv_lay = _collapsible_section()
        dl_cf_row = QHBoxLayout(); dl_cf_row.setSpacing(8)
        dl_conf_floor = _no_border(_dspin(dp["dl_detection"]["detection_confidence_floor"][0], 0.0, 1.0))
        dl_conf_floor.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "detection_confidence_floor"], [v]))
        dl_iou = _no_border(_dspin(dp["dl_detection"]["iou_threshold"], 0.0, 1.0))
        dl_iou.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "dl_detection", "iou_threshold"], v))
        dl_cf_a = QVBoxLayout(); dl_cf_a.setSpacing(4)
        dl_cf_a.addWidget(_lbl("Confidence Floor", 12, True, _C['text'])); dl_cf_a.addWidget(dl_conf_floor)
        dl_cf_b = QVBoxLayout(); dl_cf_b.setSpacing(4)
        dl_cf_b.addWidget(_lbl("IoU Threshold", 12, True, _C['text'])); dl_cf_b.addWidget(dl_iou)
        dl_cf_row.addLayout(dl_cf_a, 1); dl_cf_row.addLayout(dl_cf_b, 1)
        dl_adv_lay.addLayout(dl_cf_row)
        g.addWidget(dl_adv_btn, 3, 0, 1, 2)
        g.addWidget(dl_adv_body, 4, 0, 1, 2)
        root.addWidget(dl_card)

        sec_card = DetectorCard("Secondary Verification Model",
                                "secondary_verification" in c["detectors"])
        sec_card.enabled_signal.connect(
            lambda on: self._toggle_detector(config, update_fn, "secondary_verification", on))
        g2 = sec_card.body_layout()
        sec_model_le = _no_border(QLineEdit(dp["secondary_verification"]["model"]))
        sec_model_le.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "secondary_verification", "model"], v))
        g2.addLayout(_input_group("Model Path", _model_path_row(sec_model_le, self)), 0, 0, 1, 2)
        sec_conf = _no_border(_dspin(dp["secondary_verification"]["detection_confidence"][0], 0.0, 1.0))
        sec_conf.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "secondary_verification",
             "detection_confidence"], [v]))
        g2.addLayout(_input_group("Detection Confidence", sec_conf), 1, 0)
        sec_bpx = _no_border(_dspin(dp["secondary_verification"]["black_pixel_threshold"], 0.0, 1.0))
        sec_bpx.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "secondary_verification",
             "black_pixel_threshold"], v))
        g2.addLayout(_input_group("Black Pixel Threshold", sec_bpx), 1, 1)

        sec_adv_btn, sec_adv_body, sec_adv_lay = _collapsible_section()
        sec_sz = _no_border(QLineEdit(", ".join(map(str, dp["secondary_verification"]["image_size"]))))
        sec_sz.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "secondary_verification", "image_size"],
            [int(x) for x in v.split(",") if x.strip().isdigit()]))
        sec_adv_lay.addLayout(_input_group("Image Size (W, H)", sec_sz))
        g2.addWidget(sec_adv_btn, 2, 0, 1, 2)
        g2.addWidget(sec_adv_body, 3, 0, 1, 2)
        root.addWidget(sec_card)

        fg_card = DetectorCard("Foreground-Background (FGBG) Detection",
                               "fgbg_detection" in c["detectors"])
        fg_card.enabled_signal.connect(
            lambda on: self._toggle_detector(config, update_fn, "fgbg_detection", on))
        gf = fg_card.body_layout()
        fg_algo = _no_border(_combo(["MOG2", "FrameDifference"], dp["fgbg_detection"]["model"][0]))
        fg_algo.currentTextChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "model"], [v]))
        gf.addLayout(_input_group("Algorithm", fg_algo), 0, 0)
        fg_thresh = _no_border(_spin(dp["fgbg_detection"]["movement_threshold"], 1, 255))
        fg_thresh.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "movement_threshold"], v))
        gf.addLayout(_input_group("Movement Threshold", fg_thresh), 0, 1)
        fg_dil = _no_border(_spin(dp["fgbg_detection"]["dilate_kernel_size"], 1, 50))
        fg_dil.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "dilate_kernel_size"], v))
        gf.addLayout(_input_group("Dilate Kernel Size", fg_dil), 1, 0)
        fg_max = _no_border(_spin(dp["fgbg_detection"]["max_fgbg_candidates"], 1, 9999))
        fg_max.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "max_fgbg_candidates"], v))
        gf.addLayout(_input_group("Max Candidates", fg_max), 1, 1)

        fg_adv_btn, fg_adv_body, fg_adv_lay = _collapsible_section()

        fg_adv_row1 = QHBoxLayout(); fg_adv_row1.setSpacing(8)
        fg_ds = _no_border(_spin(dp["fgbg_detection"]["downscale_factor"], 1, 16))
        fg_ds.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "downscale_factor"], v))
        fg_wu = _no_border(_spin(dp["fgbg_detection"]["warmup_frames"], 0, 999))
        fg_wu.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "warmup_frames"], v))
        fg_ds_col = QVBoxLayout(); fg_ds_col.setSpacing(4)
        fg_ds_col.addWidget(_lbl("Downscale Factor", 12, True, _C['text'])); fg_ds_col.addWidget(fg_ds)
        fg_wu_col = QVBoxLayout(); fg_wu_col.setSpacing(4)
        fg_wu_col.addWidget(_lbl("Warmup Frames", 12, True, _C['text'])); fg_wu_col.addWidget(fg_wu)
        fg_adv_row1.addLayout(fg_ds_col, 1); fg_adv_row1.addLayout(fg_wu_col, 1)
        fg_adv_lay.addLayout(fg_adv_row1)

        fg_show_cb = QCheckBox("Show FGBG mask window")
        fg_show_cb.setChecked(dp["fgbg_detection"]["show"])
        fg_show_cb.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "show"], v))
        fg_adv_lay.addWidget(fg_show_cb)

        fg_clean_cb = QCheckBox("Clean detections")
        fg_clean_cb.setChecked(dp["fgbg_detection"]["clean_detections"])
        fg_clean_cb.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "clean_detections"], v))
        fg_adv_lay.addWidget(fg_clean_cb)

        fg_clean_body = QWidget(); fg_clean_body.setVisible(dp["fgbg_detection"]["clean_detections"])
        fg_clean_cb.toggled.connect(fg_clean_body.setVisible)
        fg_cb_lay = QVBoxLayout(fg_clean_body); fg_cb_lay.setContentsMargins(12, 4, 0, 0); fg_cb_lay.setSpacing(8)
        fg_adv_row2 = QHBoxLayout(); fg_adv_row2.setSpacing(8)
        fg_cdt = _no_border(_dspin(dp["fgbg_detection"]["clean_distance_thresh"], 0.1, 50.0))
        fg_cdt.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "clean_distance_thresh"], v))
        fg_car = _no_border(_dspin(dp["fgbg_detection"]["clean_area_ratio_thresh"], 0.0, 1.0))
        fg_car.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "detector_properties", "fgbg_detection", "clean_area_ratio_thresh"], v))
        fg_cdt_col = QVBoxLayout(); fg_cdt_col.setSpacing(4)
        fg_cdt_col.addWidget(_lbl("Clean Distance Threshold", 12, True, _C['text'])); fg_cdt_col.addWidget(fg_cdt)
        fg_car_col = QVBoxLayout(); fg_car_col.setSpacing(4)
        fg_car_col.addWidget(_lbl("Clean Area Ratio Threshold", 12, True, _C['text'])); fg_car_col.addWidget(fg_car)
        fg_adv_row2.addLayout(fg_cdt_col, 1); fg_adv_row2.addLayout(fg_car_col, 1)
        fg_cb_lay.addLayout(fg_adv_row2)
        fg_adv_lay.addWidget(fg_clean_body)

        gf.addWidget(fg_adv_btn, 2, 0, 1, 2)
        gf.addWidget(fg_adv_body, 3, 0, 1, 2)
        root.addWidget(fg_card)

        root.addStretch()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))

    @staticmethod
    def _toggle_detector(config, update_fn, key, enabled):
        dets = list(config["insect_tracking"]["detectors"])
        if enabled and key not in dets:
            dets.append(key)
        elif not enabled and key in dets:
            dets.remove(key)
        update_fn(["insect_tracking", "detectors"], dets)


class FlowerTrackingView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        c  = config["flower_tracking"]
        dp = c["detector_properties"]
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(20)
        root.addWidget(_lbl("Flower Tracking", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted(
            "Settings for identifying and tracking individual flowers for pollination analysis."))
        root.addWidget(_hsep())

        master_card = QFrame()
        master_card.setStyleSheet(f"QFrame {{ background:{_C['panel']}; border:1px solid {_C['primary']}; border-radius:10px; }}")
        m_lay = QHBoxLayout(master_card); m_lay.setContentsMargins(20, 14, 20, 14)
        m_col = QVBoxLayout(); m_col.setSpacing(2)
        m_col.addWidget(_lbl("Enable Flower Tracking", size=14, bold=True, color=_C['primary_lt']))
        m_col.addWidget(_muted("Track flowers to record pollinator visitations."))
        m_lay.addLayout(m_col, 1)
        master_toggle = ToggleSwitch(c["track"])
        master_toggle.toggled.connect(lambda v: update_fn(["flower_tracking", "track"], v))
        m_lay.addWidget(master_toggle)
        root.addWidget(master_card)

        body = QWidget(); body.setVisible(c["track"])
        master_toggle.toggled.connect(body.setVisible)
        b_lay = QVBoxLayout(body); b_lay.setSpacing(16); b_lay.setContentsMargins(0, 0, 0, 0)

        # ── Tracking Settings ─────────────────────────────────────────────────
        trk_card = CardFrame()
        trk_card.layout().addWidget(_lbl("Tracking Settings", 13, True, _C['text_bright']))
        trk_card.layout().addWidget(_hsep())

        ts_row = QHBoxLayout(); ts_row.setSpacing(8)
        shape_cb = _no_border(_combo(["Circle", "Box"], c["output_shape"]))
        shape_cb.currentTextChanged.connect(lambda v: update_fn(["flower_tracking", "output_shape"], v))
        bext = _no_border(_dspin(c["border_extension"], 0.1, 20.0))
        bext.valueChanged.connect(lambda v: update_fn(["flower_tracking", "border_extension"], v))
        ts_a = QVBoxLayout(); ts_a.setSpacing(4)
        ts_a.addWidget(_lbl("Output Shape", 12, True, _C['text'])); ts_a.addWidget(shape_cb)
        ts_b = QVBoxLayout(); ts_b.setSpacing(4)
        ts_b.addWidget(_lbl("Border Extension", 12, True, _C['text'])); ts_b.addWidget(bext)
        ts_row.addLayout(ts_a, 1); ts_row.addLayout(ts_b, 1)
        trk_card.layout().addLayout(ts_row)

        di = _no_border(_spin(c["detection_interval"], 1, 99999))
        di.valueChanged.connect(lambda v: update_fn(["flower_tracking", "detection_interval"], v))
        trk_card.layout().addLayout(_input_group("Detection Interval (frames)", di))

        trk_card.layout().addWidget(_hsep())
        mark_cb = QCheckBox("Mark insects on flower (+)")
        mark_cb.setChecked(c["mark_insects_on_flower"])
        mark_cb.toggled.connect(lambda v: update_fn(["flower_tracking", "mark_insects_on_flower"], v))
        border_cb = QCheckBox("Show border extension")
        border_cb.setChecked(c["show_border_extension"])
        border_cb.toggled.connect(lambda v: update_fn(["flower_tracking", "show_border_extension"], v))
        trk_card.layout().addWidget(mark_cb)
        trk_card.layout().addWidget(border_cb)

        trk_card.layout().addWidget(_hsep())
        fl_trk_adv_btn, fl_trk_adv_body, fl_trk_adv_lay = _collapsible_section()
        fl_labels_le = QLineEdit(", ".join(c["labels"]))
        fl_labels_le.textChanged.connect(lambda v: update_fn(["flower_tracking", "labels"],
            [x.strip() for x in v.split(",") if x.strip()]))
        fl_trk_adv_lay.addLayout(_input_group("Tracking Labels", fl_labels_le, "comma-separated"))
        fl_cls_le = QLineEdit(", ".join(map(str, c["classes"])))
        fl_cls_le.textChanged.connect(lambda v: update_fn(["flower_tracking", "classes"],
            [int(x) for x in v.split(",") if x.strip().lstrip("-").isdigit()]))
        fl_trk_adv_lay.addLayout(_input_group("YOLO Classes (comma-separated)", fl_cls_le))
        fl_pred_cb = _combo(["ConstantVelocity", "KalmanFilter"], c["prediction_method"])
        fl_pred_cb.currentTextChanged.connect(lambda v: update_fn(["flower_tracking", "prediction_method"], v))
        fl_trk_adv_lay.addLayout(_input_group("Prediction Method", fl_pred_cb))
        trk_card.layout().addWidget(fl_trk_adv_btn)
        trk_card.layout().addWidget(fl_trk_adv_body)
        b_lay.addWidget(trk_card)

        # ── Detection Model Properties ────────────────────────────────────────
        model_card = CardFrame()
        model_card.layout().addWidget(_lbl("Detection Model Properties", 13, True, _C['text_bright']))
        model_card.layout().addWidget(_hsep())

        fl_model_le = _no_border(QLineEdit(dp["model"]))
        fl_model_le.textChanged.connect(lambda v: update_fn(
            ["flower_tracking", "detector_properties", "model"], v))
        model_card.layout().addLayout(_input_group("Model Path", _model_path_row(fl_model_le, self)))

        dm_row = QHBoxLayout(); dm_row.setSpacing(8)
        fl_conf = _no_border(_dspin(dp["detection_confidence"], 0.0, 1.0))
        fl_conf.valueChanged.connect(lambda v: update_fn(
            ["flower_tracking", "detector_properties", "detection_confidence"], v))
        fl_iou = _no_border(_dspin(dp["iou_threshold"], 0.0, 1.0))
        fl_iou.valueChanged.connect(lambda v: update_fn(
            ["flower_tracking", "detector_properties", "iou_threshold"], v))
        dm_a = QVBoxLayout(); dm_a.setSpacing(4)
        dm_a.addWidget(_lbl("Confidence Threshold", 12, True, _C['text'])); dm_a.addWidget(fl_conf)
        dm_b = QVBoxLayout(); dm_b.setSpacing(4)
        dm_b.addWidget(_lbl("IoU Threshold", 12, True, _C['text'])); dm_b.addWidget(fl_iou)
        dm_row.addLayout(dm_a, 1); dm_row.addLayout(dm_b, 1)
        model_card.layout().addLayout(dm_row)

        fl_fp16 = QCheckBox("Use FP16 (CUDA only)")
        fl_fp16.setChecked(dp["use_fp16"])
        fl_fp16.toggled.connect(lambda v: update_fn(
            ["flower_tracking", "detector_properties", "use_fp16"], v))
        model_card.layout().addWidget(fl_fp16)

        model_card.layout().addWidget(_hsep())
        fl_mdl_adv_btn, fl_mdl_adv_body, fl_mdl_adv_lay = _collapsible_section()
        fl_sz_le = _no_border(QLineEdit(", ".join(map(str, dp["image_size"]))))
        fl_sz_le.textChanged.connect(lambda v: update_fn(
            ["flower_tracking", "detector_properties", "image_size"],
            [int(x) for x in v.split(",") if x.strip().isdigit()]))
        fl_mdl_adv_lay.addLayout(_input_group("Image Size (W, H)", fl_sz_le))
        model_card.layout().addWidget(fl_mdl_adv_btn)
        model_card.layout().addWidget(fl_mdl_adv_body)
        b_lay.addWidget(model_card)

        root.addWidget(body)
        root.addStretch()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))


class AdvancedView(QWidget):

    def __init__(self, config: dict, update_fn, parent=None):
        super().__init__(parent)
        sf = config["insect_tracking"]["spatial_filtering"]
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        root = QVBoxLayout(inner); root.setContentsMargins(32, 28, 32, 32); root.setSpacing(20)
        root.addWidget(_lbl("Advanced System Settings", size=18, bold=True, color=_C['text_bright']))
        root.addWidget(_muted("Hardware acceleration, spatial filtering, and edge-case behaviour."))
        root.addWidget(_hsep())

        # ── Spatial Filtering ─────────────────────────────────────────────────
        sf_card = CardFrame()
        sf_card.layout().addWidget(_lbl("Spatial Filtering", 13, True, _C['text_bright']))
        sf_card.layout().addWidget(_hsep())

        inc_cb = QCheckBox("Use Include Zone")
        inc_cb.setChecked(sf["use_include_zone"])
        inc_cb.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "spatial_filtering", "use_include_zone"], v))
        sf_card.layout().addWidget(inc_cb)

        inc_coord_w = QWidget(); inc_coord_w.setVisible(sf["use_include_zone"])
        inc_cb.toggled.connect(inc_coord_w.setVisible)
        inc_coord_le = QLineEdit(", ".join(map(str, sf["include_zone_coord"])))
        inc_coord_le.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "spatial_filtering", "include_zone_coord"],
            [int(x) for x in v.split(",") if x.strip().lstrip("-").isdigit()]))
        inc_inner = QVBoxLayout(inc_coord_w); inc_inner.setContentsMargins(12, 0, 0, 0)
        inc_inner.addLayout(_input_group("Include Coords [x1,y1,x2,y2]", inc_coord_le))
        sf_card.layout().addWidget(inc_coord_w)

        sf_card.layout().addWidget(_hsep())

        exc_cb = QCheckBox("Use Exclude Zone")
        exc_cb.setChecked(sf["use_exclude_zone"])
        exc_cb.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "spatial_filtering", "use_exclude_zone"], v))
        sf_card.layout().addWidget(exc_cb)

        exc_coord_w = QWidget(); exc_coord_w.setVisible(sf["use_exclude_zone"])
        exc_cb.toggled.connect(exc_coord_w.setVisible)
        exc_coord_le = QLineEdit(", ".join(map(str, sf["exclude_zone_coord"])))
        exc_coord_le.textChanged.connect(lambda v: update_fn(
            ["insect_tracking", "spatial_filtering", "exclude_zone_coord"],
            [int(x) for x in v.split(",") if x.strip().lstrip("-").isdigit()]))
        exc_inner = QVBoxLayout(exc_coord_w); exc_inner.setContentsMargins(12, 0, 0, 0)
        exc_inner.addLayout(_input_group("Exclude Coords [x1,y1,x2,y2]", exc_coord_le))
        sf_card.layout().addWidget(exc_coord_w)

        root.addWidget(sf_card)

        hw_card = CardFrame()
        hw_card.layout().addWidget(_lbl("Hardware Acceleration", 13, True, _C['text_bright']))
        hw_card.layout().addWidget(_hsep())
        dev_combo = _combo(["auto", "cuda:0", "cpu", "mps"], config["source"]["device"])
        dev_combo.currentTextChanged.connect(lambda v: update_fn(["source", "device"], v))
        hw_card.layout().addLayout(_input_group("Target Compute Device", dev_combo,
                                                 "auto = CUDA > MPS > CPU"))
        hw_card.layout().addWidget(
            _muted("Using 'auto' selects the best available GPU before falling back to CPU."))
        root.addWidget(hw_card)

        ea = config["insect_tracking"]["edge_analysis"]
        edge_card = CardFrame()
        edge_card.layout().addWidget(_lbl("Edge Behaviour Analysis", 13, True, _C['text_bright']))
        edge_card.layout().addWidget(_hsep())
        edge_grid = QGridLayout(); edge_grid.setSpacing(10)
        ep = _spin(ea["edge_pixels"], 1, 999)
        ep.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "edge_analysis", "edge_pixels"], v))
        mo = _spin(ea["max_edge_occlusions"], 0, 9999)
        mo.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "edge_analysis", "max_edge_occlusions"], v))
        ep_col = QVBoxLayout(); ep_col.setSpacing(4)
        ep_col.addWidget(_lbl("Edge Pixels Threshold", 12, True, _C['text'])); ep_col.addWidget(ep)
        mo_col = QVBoxLayout(); mo_col.setSpacing(4)
        mo_col.addWidget(_lbl("Max Edge Occlusions", 12, True, _C['text'])); mo_col.addWidget(mo)
        edge_grid.addLayout(ep_col, 0, 0); edge_grid.addLayout(mo_col, 0, 1)
        edge_card.layout().addLayout(edge_grid)
        cont_cb = QCheckBox("Enable Continuous Edge Analysis")
        cont_cb.setChecked(ea["continious_analysis"])
        cont_cb.toggled.connect(lambda v: update_fn(
            ["insect_tracking", "edge_analysis", "continious_analysis"], v))
        edge_card.layout().addWidget(cont_cb)

        edge_card.layout().addWidget(_hsep())
        ea_adv_btn, ea_adv_body, ea_adv_lay = _collapsible_section()
        ea_cvtj = _spin(ea["compressed_video_time_jump"], 1, 9999)
        ea_cvtj.valueChanged.connect(lambda v: update_fn(
            ["insect_tracking", "edge_analysis", "compressed_video_time_jump"], v))
        ea_adv_lay.addLayout(_input_group("Compressed Video Time Jump (frames)", ea_cvtj,
                                          "Frame gap for edge detection in compressed video"))
        edge_card.layout().addWidget(ea_adv_btn)
        edge_card.layout().addWidget(ea_adv_body)
        root.addWidget(edge_card)

        root.addStretch()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scrolled(inner))


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING DASHBOARD (QDialog)
# ═══════════════════════════════════════════════════════════════════════════════

class TrackingDashboard(QDialog):

    _RE_PROGRESS = re.compile(r"\]\s*([\d.]+)%")
    _RE_ACTIVE    = re.compile(r"(\d+)\s+active tracks")
    _RE_SAVED     = re.compile(r"(\d+)\s+saved tracks")
    _RE_FLOWERS   = re.compile(r"(\d+)\s+flowers")
    _RE_FPS       = re.compile(r"Processing FPS:\s*([\d.]+)")

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config    = config
        self._process   = None
        self._tmp_yaml  = None
        self._frame_pipe: str | None = None
        self._frame_timer: QTimer | None = None
        self._last_frame_mtime: float = 0.0

        show_video = config["output"]["show"]
        self.setWindowTitle("Polytrack – Live Tracking")
        self.setMinimumSize(1100 if show_video else 860, 680)
        self.resize(1280 if show_video else 900, 720)
        self.setModal(True)
        self.setStyleSheet(f"background:{_C['surface']};")

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(56)
        hdr.setStyleSheet(f"background:{_C['surface']}; border-bottom:1px solid {_C['border']};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20, 0, 20, 0); hl.setSpacing(12)
        self._mode_lbl = _lbl(
            "Live Video Tracking" if show_video else "Headless Tracking Mode",
            size=15, bold=True, color=_C['text_bright'])
        hl.addWidget(self._mode_lbl)
        hl.addStretch()
        self._status_chip = QLabel("● Initialising")
        self._status_chip.setStyleSheet(
            f"background:rgba(59,130,246,0.2); color:{_C['focus']};"
            "border-radius:10px; padding:3px 12px; font-size:12px; font-weight:600;")
        hl.addWidget(self._status_chip)
        self._close_btn = QPushButton("✕  Stop & Close")
        self._close_btn.setStyleSheet(
            "QPushButton { background:rgba(220,38,38,0.15); color:#FCA5A5;"
            "border:1px solid rgba(220,38,38,0.4); border-radius:6px;"
            "padding:5px 16px; font-weight:600; font-size:12px; }"
            "QPushButton:hover { background:rgba(220,38,38,0.3); }"
            "QPushButton:pressed { background:rgba(220,38,38,0.5); }")
        self._close_btn.clicked.connect(self._on_close)
        hl.addWidget(self._close_btn)
        root.addWidget(hdr)

        body = self._build_video_body() if show_video else self._build_headless_body()
        root.addWidget(body, 1)
        QTimer.singleShot(200, self._start)

    def _build_headless_body(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background:{_C['surface']};")
        lay = QVBoxLayout(w); lay.setContentsMargins(32, 24, 32, 24); lay.setSpacing(20)

        stats_row = QHBoxLayout(); stats_row.setSpacing(16)
        self._stat_cards = {}
        for key, title, val, color, icon in [
            ("visual",  "Visual Output",    "DISABLED", _C['muted'],      "🖥"),
            ("flowers", "Detected Flowers", "0",        _C['pink'],       "🌸"),
            ("insects", "Active Insects",   "0",        _C['primary_lt'], "🐝"),
        ]:
            card = self._make_stat_card(title, val, color, icon)
            self._stat_cards[key] = card["val_lbl"]
            stats_row.addWidget(card["frame"], 1)
        lay.addLayout(stats_row)

        ph = QHBoxLayout()
        ph.addWidget(_lbl("BATCH PROCESSING PROGRESS", 10, True, _C['text_dim']))
        ph.addStretch()
        self._pct_lbl = _lbl("0%", 14, True, _C['focus'])
        ph.addWidget(self._pct_lbl)
        lay.addLayout(ph)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000); self._progress.setValue(0)
        self._progress.setTextVisible(False); self._progress.setFixedHeight(14)
        lay.addWidget(self._progress)
        lay.addWidget(self._make_log_widget(), 1)
        return w

    def _build_video_body(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle {{ background:{_C['border']}; }}")
        splitter.setHandleWidth(4)

        left = QWidget(); left.setStyleSheet("background:#000;")
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(0)
        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_lbl.setText("Waiting for first frame…")
        self._video_lbl.setStyleSheet(f"background:#000; color:{_C['muted']}; font-size:13px;")
        ll.addWidget(self._video_lbl, 1)
        vid_bar = QWidget(); vid_bar.setFixedHeight(36)
        vid_bar.setStyleSheet(f"background:{_C['surface']};")
        vb = QHBoxLayout(vid_bar); vb.setContentsMargins(14, 0, 14, 0)
        self._frame_lbl = _lbl("Frame: — / —", 10, False, _C['placeholder'])
        vb.addWidget(self._frame_lbl); vb.addStretch()
        ll.addWidget(vid_bar)
        splitter.addWidget(left)

        right = QWidget(); right.setFixedWidth(380)
        right.setStyleSheet(f"background:{_C['panel']};")
        rl = QVBoxLayout(right); rl.setContentsMargins(16, 16, 16, 16); rl.setSpacing(14)

        ph = QHBoxLayout()
        ph.addWidget(_lbl("Progress", 11, True, _C['text_dim'])); ph.addStretch()
        self._pct_lbl = _lbl("0%", 13, True, _C['focus'])
        ph.addWidget(self._pct_lbl)
        rl.addLayout(ph)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000); self._progress.setValue(0)
        self._progress.setTextVisible(False); self._progress.setFixedHeight(10)
        rl.addWidget(self._progress)
        self._frame_lbl2 = _lbl("Frame 0 of 0", 10, False, _C['placeholder'])
        rl.addWidget(self._frame_lbl2)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{_C['border']};"); rl.addWidget(sep)

        sg = QGridLayout(); sg.setSpacing(10)
        self._stat_cards = {}
        for i, (key, title, val, color) in enumerate([
            ("insects", "Active Insects", "0", _C['focus']),
            ("saved",   "Saved Tracks",   "0", "#38BDF8"),
            ("flowers", "Flowers",        "0", _C['pink']),
            ("fps",     "Proc. FPS",      "—", _C['primary_lt']),
        ]):
            card = self._make_stat_card(title, val, color, "", small=True)
            self._stat_cards[key] = card["val_lbl"]
            sg.addWidget(card["frame"], i // 2, i % 2)
        rl.addLayout(sg)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{_C['border']};"); rl.addWidget(sep2)
        rl.addWidget(self._make_log_widget(), 1)
        splitter.addWidget(right)
        splitter.setSizes([820, 380])
        return splitter

    def _make_stat_card(self, title: str, val: str, color: str,
                         icon: str, small: bool = False) -> dict:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background:{_C['panel']}; border:1px solid {_C['border']}; border-radius:12px; }}")
        frame.setMinimumHeight(60 if small else 90)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(12 if small else 16, 10, 12 if small else 16, 10); fl.setSpacing(4)
        if icon:
            hr = QHBoxLayout(); hr.addWidget(_lbl(icon, 18, False, color)); hr.addStretch()
            fl.addLayout(hr)
        fl.addWidget(_lbl(title, 10, True, _C['text_dim']))
        val_lbl = _lbl(val, 28 if not small else 22, True, color)
        fl.addWidget(val_lbl)
        return {"frame": frame, "val_lbl": val_lbl}

    def _make_log_widget(self) -> QWidget:
        container = QFrame()
        container.setStyleSheet(f"QFrame {{ background:{_C['surface']}; border:1px solid {_C['border']}; border-radius:10px; }}")
        cl = QVBoxLayout(container); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        log_hdr = QWidget(); log_hdr.setFixedHeight(32)
        log_hdr.setStyleSheet(
            f"background:{_C['surface']}; border-bottom:1px solid {_C['border']}; border-radius:10px 10px 0 0;")
        lhlay = QHBoxLayout(log_hdr); lhlay.setContentsMargins(12, 0, 12, 0)
        lhlay.addWidget(_lbl("⬤  LIVE EXECUTION LOGS", 9, True, _C['placeholder']))
        lhlay.addStretch()
        cl.addWidget(log_hdr)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFont(QFont("Courier New", 10))
        self._log_box.setStyleSheet(
            f"QTextEdit {{ background:{_C['surface']}; color:{_C['text_dim']}; border:none; border-radius:0 0 10px 10px; padding:8px; }}")
        cl.addWidget(self._log_box, 1)
        return container

    def _start(self):
        show_video = self._config["output"]["show"]
        cfg_to_write = _config_for_yaml(self._config)
        if show_video:
            cfg_to_write["output"]["show"] = False
        try:
            self._tmp_fd, self._tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="polytrack_")
            os.close(self._tmp_fd)
            with open(self._tmp_yaml, "w", encoding="utf-8") as fh:
                yaml.dump(cfg_to_write, fh, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            self._log(f"[ERROR] Could not write temp config: {exc}", _C['error']); return

        main_py = str(Path(__file__).parent / "main.py")
        if not Path(main_py).exists():
            self._log(f"[ERROR] main.py not found at: {main_py}", _C['error']); return

        args = [main_py, "--config", self._tmp_yaml]
        if show_video:
            _fd, self._frame_pipe = tempfile.mkstemp(suffix=".jpg", prefix="polytrack_frame_")
            os.close(_fd)
            args += ["--frame-pipe", self._frame_pipe]
            self._frame_timer = QTimer(self)
            self._frame_timer.timeout.connect(self._poll_frame)
            self._frame_timer.start(66)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(str(Path(__file__).parent))
        self._process.readyRead.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        self._set_status("● Running", _C['primary_lt'], f"rgba(5,150,105,0.2)")
        self._log(f"[INFO] Launching: python {Path(main_py).name}", _C['focus'])
        self._log(f"[INFO] Config:    {self._tmp_yaml}", _C['text_dim'])
        self._log(f"[INFO] Source:    {self._config['directories']['source']}", _C['text_dim'])
        self._process.start(sys.executable, args)

    def _on_stdout(self):
        if not self._process: return
        raw = bytes(self._process.readAll()).decode("utf-8", errors="replace")
        for line in re.split(r"[\r\n]+", raw):
            line = line.strip()
            if not line: continue
            self._parse_progress_line(line)
            self._log_line(line)

    def _parse_progress_line(self, line: str):
        m = self._RE_PROGRESS.search(line)
        if m:
            pct = float(m.group(1))
            self._progress.setValue(int(pct * 10))
            self._pct_lbl.setText(f"{pct:.1f}%")
        m = self._RE_ACTIVE.search(line)
        if m and "insects" in self._stat_cards:
            self._stat_cards["insects"].setText(m.group(1))
        m = self._RE_SAVED.search(line)
        if m and "saved" in self._stat_cards:
            self._stat_cards["saved"].setText(m.group(1))
        m = self._RE_FLOWERS.search(line)
        if m and "flowers" in self._stat_cards:
            self._stat_cards["flowers"].setText(m.group(1))
        m = self._RE_FPS.search(line)
        if m and "fps" in self._stat_cards:
            self._stat_cards["fps"].setText(f"{float(m.group(1)):.1f}")
        m_f = re.search(r"(\d+)/(\d+)\s+frames", line)
        if m_f:
            txt = f"Frame {m_f.group(1)} of {m_f.group(2)}"
            if hasattr(self, "_frame_lbl"):  self._frame_lbl.setText(txt)
            if hasattr(self, "_frame_lbl2"): self._frame_lbl2.setText(txt)

    def _log_line(self, line: str):
        if "ERROR" in line or "error" in line.lower(): col = _C['error']
        elif "WARNING" in line or "warning" in line.lower(): col = _C['warn']
        elif line.startswith("[INFO]") or "INFO" in line: col = _C['text']
        elif "%" in line and "[" in line: col = _C['focus']
        else: col = _C['text_dim']
        self._log(line, col)

    def _poll_frame(self):
        if not self._frame_pipe or not Path(self._frame_pipe).exists(): return
        try:
            mtime = Path(self._frame_pipe).stat().st_mtime
        except OSError:
            return
        if mtime <= self._last_frame_mtime: return
        self._last_frame_mtime = mtime
        pix = QPixmap(self._frame_pipe)
        if pix.isNull(): return
        pix = pix.scaled(self._video_lbl.size(),
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self._video_lbl.setPixmap(pix)

    def _stop_frame_timer(self):
        if self._frame_timer:
            self._frame_timer.stop(); self._frame_timer = None

    def _on_finished(self, exit_code: int, _status):
        self._stop_frame_timer()
        self._progress.setValue(1000); self._pct_lbl.setText("100%")
        if exit_code == 0:
            self._set_status("● Complete", _C['primary_lt'], "rgba(5,150,105,0.2)")
            self._log("[INFO] Processing complete.", _C['primary_lt'])
        else:
            self._set_status("● Failed", "#FCA5A5", "rgba(220,38,38,0.2)")
            self._log(f"[ERROR] Exited with code {exit_code}.", _C['error'])
        self._close_btn.setText("Close")

    def _on_error(self, error):
        names = {QProcess.ProcessError.FailedToStart: "FailedToStart",
                 QProcess.ProcessError.Crashed: "Crashed"}
        self._log(f"[ERROR] QProcess error: {names.get(error, str(error))}", _C['error'])
        self._set_status("● Error", "#FCA5A5", "rgba(220,38,38,0.2)")

    def _on_close(self):
        self._stop_frame_timer()
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate(); self._process.waitForFinished(2000)
        for tmp in (self._tmp_yaml, self._frame_pipe):
            if tmp and Path(tmp).exists():
                try: os.unlink(tmp)
                except OSError: pass
        self.close()

    def _log(self, text: str, color: str | None = None):
        if hasattr(self, "_log_box"):
            self._log_box.append(f'<span style="color:{color or _C["text_dim"]};">{text}</span>')
            self._log_box.moveCursor(QTextCursor.MoveOperation.End)

    def _set_status(self, text: str, text_color: str, bg: str):
        self._status_chip.setText(text)
        self._status_chip.setStyleSheet(
            f"background:{bg}; color:{text_color}; border-radius:10px;"
            "padding:3px 12px; font-size:12px; font-weight:600;")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("📁", "Paths & Directories"),
    ("🎬", "Input / Output Config"),
    ("🐝", "Insect: General"),
    ("🔍", "Insect: Detectors"),
    ("🌸", "Flower Tracking"),
    ("⚙️", "Advanced Settings"),
]


class PolytrackMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polytrack – Environment Pollinator Tracking")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 820)
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        self._theme_name = "dark" if _C is _DARK or _C['bg'] == _DARK['bg'] else "light"
        self._build_ui()

    # ── UI construction (called once, and again on theme switch) ──────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout(); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        vdiv = QFrame(); vdiv.setFrameShape(QFrame.Shape.VLine)
        vdiv.setStyleSheet(f"color:{_C['border']};")
        body.addWidget(vdiv)
        body.addWidget(self._build_content(), 1)

        body_w = QWidget(); body_w.setStyleSheet(f"background:{_C['bg']};")
        body_w.setLayout(body)
        root.addWidget(body_w, 1)

        self.statusBar().setStyleSheet(
            f"background:{_C['surface']}; color:{_C['placeholder']};"
            f"border-top:1px solid {_C['border']}; font-size:11px;")
        self.statusBar().showMessage("Ready.  Select a video folder and config file to begin.")

    def _build_header(self) -> QWidget:
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet(f"background:{_C['surface']}; border-bottom:1px solid {_C['border']};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(24, 0, 24, 0); hl.setSpacing(12)

        logo_col = QVBoxLayout(); logo_col.setSpacing(1)
        logo_col.addWidget(_lbl("Polytrack", 20, True, _C['text_bright']))
        logo_col.addWidget(_lbl("Environment Pollinator Tracking", 9, False, _C['primary_lt']))
        hl.addLayout(logo_col)
        hl.addStretch()

        chip = QLabel("v5.0")
        chip.setStyleSheet(
            f"background:rgba(5,150,105,0.2); color:{_C['primary_lt']};"
            "border-radius:10px; padding:2px 10px; font-size:11px; font-weight:600;")
        hl.addWidget(chip)
        hl.addSpacing(8)

        # Theme toggle button
        is_dark = _C['bg'] == _DARK['bg']
        theme_icon = "☀  Light" if is_dark else "🌙  Dark"
        self._theme_btn = QPushButton(theme_icon)
        self._theme_btn.setFixedHeight(34)
        self._theme_btn.setMinimumWidth(96)
        self._theme_btn.setStyleSheet(
            f"QPushButton {{ background:{_C['panel']}; color:{_C['text_dim']}; border:1px solid {_C['border']}; border-radius:14px; padding:4px 14px; font-size:12px; }} QPushButton:hover {{ border-color:{_C['focus']}; color:{_C['text']}; }}")
        self._theme_btn.clicked.connect(self._toggle_theme)
        hl.addWidget(self._theme_btn)
        hl.addSpacing(8)

        save_btn = QPushButton("  Save Config")
        save_btn.setFixedHeight(36); save_btn.setMinimumWidth(130)
        save_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{_C['text']}; border:1.5px solid {_C['border']}; border-radius:6px; padding:6px 18px; font-weight:600; font-size:13px; }} QPushButton:hover {{ background:{_C['panel']}; border-color:{_C['muted']}; color:{_C['text_bright']}; }} QPushButton:pressed {{ background:{_C['border']}; }}")
        save_btn.clicked.connect(self._save_config)
        hl.addWidget(save_btn)

        start_btn = QPushButton("▶   Start Tracking")
        start_btn.setFixedHeight(40); start_btn.setMinimumWidth(160)
        start_btn.setStyleSheet(
            f"QPushButton {{ background:{_C['primary']}; color:#FFFFFF; border:none; border-radius:8px; padding:8px 24px; font-weight:700; font-size:14px; }} QPushButton:hover {{ background:{_C['primary_hv']}; }} QPushButton:pressed {{ background:{_C['primary_hv']}; }}")
        start_btn.clicked.connect(self._start_tracking)
        hl.addWidget(start_btn)
        return hdr

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget(); sidebar.setFixedWidth(240)
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet(f"QWidget#sidebar {{ background:{_C['surface']}; }}")
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(0, 16, 0, 16); sl.setSpacing(0)

        sec = QLabel("CONFIGURATION")
        sec.setContentsMargins(24, 0, 0, 10)
        sec.setStyleSheet(
            f"color:{_C['text_dim']}; letter-spacing:1.2px; font-size:10px;"
            "font-weight:700; background:transparent;")
        sl.addWidget(sec)

        self._nav = QListWidget()
        self._nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for icon, text in NAV_ITEMS:
            item = QListWidgetItem(f"  {icon}  {text}")
            item.setSizeHint(QSize(0, 44))
            self._nav.addItem(item)
        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(lambda i: self._stack.setCurrentIndex(i))
        sl.addWidget(self._nav, 1)
        return sidebar

    def _build_content(self) -> QStackedWidget:
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background:transparent;")
        prev_row = self._nav.currentRow() if hasattr(self, "_nav") else 0
        for view in [
            PathsView(self._config, self._update),
            IOConfigView(self._config, self._update),
            InsectGeneralView(self._config, self._update),
            InsectDetectorsView(self._config, self._update),
            FlowerTrackingView(self._config, self._update),
            AdvancedView(self._config, self._update),
        ]:
            self._stack.addWidget(view)
        self._stack.setCurrentIndex(prev_row)
        return self._stack

    # ── Theme toggle ──────────────────────────────────────────────────────────

    def _toggle_theme(self):
        current_row = self._nav.currentRow() if hasattr(self, "_nav") else 0
        new_theme = "light" if _C['bg'] == _DARK['bg'] else "dark"
        _set_theme(new_theme)
        QApplication.instance().setStyleSheet(build_qss(_C))
        # Rebuild the UI so all inline-styled widgets get the new colours
        self._build_ui()
        self._nav.setCurrentRow(current_row)
        self._stack.setCurrentIndex(current_row)

    # ── Config helpers ────────────────────────────────────────────────────────

    def _update(self, path: list, value) -> None:
        _deep_set(self._config, path, value)

    def _save_config(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Config", str(Path.home()), "YAML files (*.yaml *.yml)")
        if not dest: return
        try:
            with open(dest, "w", encoding="utf-8") as fh:
                yaml.dump(_config_for_yaml(self._config), fh,
                          default_flow_style=False, allow_unicode=True)
            self.statusBar().showMessage(f"Config saved → {dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _start_tracking(self) -> None:
        if not self._config["directories"]["source"]:
            QMessageBox.warning(self, "Missing input",
                                "Please set a source path in Paths & Directories.")
            return
        TrackingDashboard(self._config, self).exec()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Detect system theme before building any widgets
    _set_theme(_detect_system_theme())
    app.setStyleSheet(build_qss(_C))

    win = PolytrackMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
