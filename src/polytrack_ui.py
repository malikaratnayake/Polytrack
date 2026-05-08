"""
polytrack_ui.py
===============
PySide6 graphical interface for Polytrack – insect tracking and pollination
monitoring.  Designed for non-technical ecologists who need to select videos,
configure the YAML config, run processing, and inspect results without touching
the command line.

Screens
-------
MainMenuScreen       – Landing page with logo and navigation.
ProcessVideosScreen  – Pick input / config / output paths and run main.py.
ConfigureScreen      – Browse and edit key config parameters.
ResultsScreen        – Scan an output folder and open result files.

Run
---
    python polytrack_ui.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
from pathlib import Path

import yaml

from PySide6.QtCore import (
    Qt, QProcess, QTimer, QPointF, QRectF, QSize,
)
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPainterPath,
    QPen, QRadialGradient, QBrush, QPixmap, QIcon, QDesktopServices,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog,
    QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QScrollArea, QSizePolicy, QFrame, QSplitter, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QGroupBox,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
YELLOW   = QColor("#FACC15")
YELLOW_D = QColor("#CA9F0A")
BG_DARK  = QColor("#0D0D0D")
BG_MID   = QColor("#1A1A2E")
GLASS_BG = QColor(20, 20, 40, 180)
GLASS_BD = QColor(250, 204, 21, 60)
TEXT_PRI = QColor("#F8F8F2")
TEXT_SEC = QColor("#A0A0B0")
ACCENT   = YELLOW
SUCCESS  = QColor("#22C55E")
WARNING  = QColor("#F97316")
ERROR    = QColor("#EF4444")

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _monospace_font(size: int = 11) -> QFont:
    f = QFont("Courier New")
    f.setPointSize(size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def _ui_font(size: int = 13, bold: bool = False) -> QFont:
    f = QFont("Helvetica Neue")
    f.setPointSize(size)
    f.setBold(bold)
    f.setStyleHint(QFont.StyleHint.SansSerif)
    return f


# ---------------------------------------------------------------------------
# Background mixin – radial gradient + perspective grid
# ---------------------------------------------------------------------------

class BackgroundMixin:
    """
    Mixin for QWidget subclasses.  Override paintEvent to draw the shared
    dark radial-gradient backdrop with a perspective-vanishing-point grid.
    """

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # --- radial gradient background ---
        grad = QRadialGradient(w / 2, h / 3, max(w, h) * 0.85)
        grad.setColorAt(0.0, QColor("#1B1B3A"))
        grad.setColorAt(0.5, QColor("#0F0F1F"))
        grad.setColorAt(1.0, QColor("#070710"))
        p.fillRect(0, 0, w, h, grad)

        # --- perspective grid (lines converge to horizon point) ---
        p.setPen(QPen(QColor(250, 204, 21, 18), 1))
        cx, hy = w / 2, h * 0.42          # vanishing point

        # vertical fan lines
        n_lines = 22
        spread  = w * 1.1
        for i in range(n_lines + 1):
            x_bot = -spread / 2 + spread * i / n_lines
            p.drawLine(QPointF(cx, hy), QPointF(x_bot, float(h)))

        # horizontal parallels
        n_horiz = 14
        for j in range(1, n_horiz + 1):
            frac = j / n_horiz
            y = hy + (h - hy) * frac
            t = frac ** 1.6
            x0 = cx - spread / 2 * t
            x1 = cx + spread / 2 * t
            p.drawLine(QPointF(x0, y), QPointF(x1, y))

        super().paintEvent(event)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GlassPanel
# ---------------------------------------------------------------------------

class GlassPanel(QFrame):
    """
    Semi-transparent dark card with a yellow accent border.
    Drop this widget in place of a plain QFrame for a consistent look.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        # fill
        p.setBrush(QBrush(GLASS_BG))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 12, 12)
        # border
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(GLASS_BD, 1.5))
        p.drawRoundedRect(r.adjusted(0.75, 0.75, -0.75, -0.75), 12, 12)
        super().paintEvent(event)


# ---------------------------------------------------------------------------
# PolyButton – slanted parallelogram push-button
# ---------------------------------------------------------------------------

class PolyButton(QPushButton):
    """
    Custom push-button rendered as a slanted parallelogram (arcade style).

    Parameters
    ----------
    text : str
    primary : bool
        True → yellow fill, black text.  False → dark fill, yellow text.
    small : bool
        Reduce padding for compact layouts.
    """

    SKEW = 14   # horizontal skew in pixels

    def __init__(self, text: str = "", primary: bool = True,
                 small: bool = False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.small   = small
        self._hovered = False
        self._pressed = False
        h = 38 if small else 48
        self.setMinimumHeight(h)
        self.setMinimumWidth(120 if small else 160)
        self.setFont(_ui_font(11 if small else 13, bold=True))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    # --- mouse state tracking ---
    def enterEvent(self, e):  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._hovered = False
        self._pressed = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):  # noqa: N802
        self._pressed = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(e)

    # --- paint ---
    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        sk = self.SKEW

        # build parallelogram path
        path = QPainterPath()
        path.moveTo(sk,   0)
        path.lineTo(w,    0)
        path.lineTo(w-sk, h)
        path.lineTo(0,    h)
        path.closeSubpath()

        # colours
        if self.primary:
            base = YELLOW_D if self._pressed else (QColor("#E8B800") if self._hovered else YELLOW)
            text_col = QColor("#0D0D0D")
        else:
            base = QColor(40, 40, 60, 220) if self._pressed else (
                   QColor(50, 50, 80, 200) if self._hovered else QColor(25, 25, 45, 200))
            text_col = YELLOW

        p.setBrush(QBrush(base))
        border_col = YELLOW if not self.primary else QColor(0, 0, 0, 60)
        p.setPen(QPen(border_col, 1.5))
        p.drawPath(path)

        # text
        p.setPen(QPen(text_col))
        p.setFont(self.font())
        text_rect = path.boundingRect().adjusted(sk / 2, 0, -sk / 2, 0)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) + self.SKEW * 2 + 40
        h      = 38 if self.small else 48
        return QSize(max(text_w, 160 if not self.small else 120), h)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _label(text: str, size: int = 12, bold: bool = False,
           color: QColor = TEXT_PRI) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(_ui_font(size, bold))
    pal = lbl.palette()
    pal.setColor(lbl.foregroundRole(), color)
    lbl.setPalette(pal)
    lbl.setStyleSheet(f"color: {color.name()};")
    return lbl


def _section_title(text: str) -> QLabel:
    lbl = _label(text, size=14, bold=True, color=YELLOW)
    lbl.setContentsMargins(0, 8, 0, 4)
    return lbl


def _style_input(widget: QWidget) -> QWidget:
    widget.setStyleSheet("""
        background-color: rgba(20,20,40,220);
        color: #F8F8F2;
        border: 1px solid rgba(250,204,21,80);
        border-radius: 6px;
        padding: 4px 8px;
        selection-background-color: #FACC15;
        selection-color: #0D0D0D;
    """)
    return widget


def _path_row(label_text: str, placeholder: str,
              pick_dir: bool = False, pick_file_filter: str = "") -> tuple[QHBoxLayout, QLineEdit]:
    """Return a layout row and the LineEdit for a file/folder picker."""
    row = QHBoxLayout()
    row.setSpacing(8)
    le = QLineEdit()
    le.setPlaceholderText(placeholder)
    _style_input(le)
    le.setMinimumHeight(34)

    btn = PolyButton("Browse", primary=False, small=True)
    btn.setMinimumWidth(90)

    if pick_dir:
        def _pick():
            d = QFileDialog.getExistingDirectory(None, label_text, str(Path.home()))
            if d:
                le.setText(d)
        btn.clicked.connect(_pick)
    else:
        def _pick():
            f, _ = QFileDialog.getOpenFileName(None, label_text, str(Path.home()), pick_file_filter)
            if f:
                le.setText(f)
        btn.clicked.connect(_pick)

    row.addWidget(le, 1)
    row.addWidget(btn)
    return row, le


# ---------------------------------------------------------------------------
# Screen 1 – Main Menu
# ---------------------------------------------------------------------------

class MainMenuScreen(BackgroundMixin, QWidget):

    def __init__(self, navigate, parent=None):
        super().__init__(parent)
        self._navigate = navigate
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 40, 60, 40)
        root.setSpacing(0)

        # ── Logo ────────────────────────────────────────────────────────────
        logo_panel = GlassPanel()
        logo_layout = QVBoxLayout(logo_panel)
        logo_layout.setContentsMargins(30, 28, 30, 28)

        title_lbl = QLabel("POLY<span style='color:#FACC15'>TRACK</span>")
        title_lbl.setTextFormat(Qt.TextFormat.RichText)
        title_f = QFont("Helvetica Neue")
        title_f.setPointSize(52)
        title_f.setBold(True)
        title_f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
        title_lbl.setFont(title_f)
        title_lbl.setStyleSheet("color: #F8F8F2; background: transparent;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub_lbl = _label(
            "Insect Tracking & Pollination Monitoring  ·  v5.0",
            size=13, color=TEXT_SEC)
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_layout.addWidget(title_lbl)
        logo_layout.addWidget(sub_lbl)

        root.addWidget(logo_panel)
        root.addSpacing(32)

        # ── Navigation + Info side-by-side ──────────────────────────────────
        mid = QHBoxLayout()
        mid.setSpacing(24)

        # Navigation buttons
        nav_panel = GlassPanel()
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(28, 28, 28, 28)
        nav_layout.setSpacing(14)
        nav_layout.addWidget(_label("What would you like to do?", size=14, bold=True))
        nav_layout.addSpacing(6)

        buttons = [
            ("Process Videos",   "process",   True),
            ("Configure",        "configure", True),
            ("View Results",     "results",   True),
        ]
        for text, key, primary in buttons:
            btn = PolyButton(text, primary=primary)
            btn.clicked.connect(lambda _=False, k=key: self._navigate(k))
            nav_layout.addWidget(btn)

        nav_layout.addStretch()

        # Quick-start info card
        info_panel = GlassPanel()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(24, 24, 24, 24)
        info_layout.setSpacing(10)
        info_layout.addWidget(_section_title("Quick Start"))

        steps = [
            ("1", "Configure",      "Point Polytrack to your detection models and set your thresholds."),
            ("2", "Process Videos", "Select your video folder, config file, and output location."),
            ("3", "View Results",   "Open your output folder to inspect tracks and visualisations."),
        ]
        for num, title, desc in steps:
            row = QHBoxLayout()
            num_lbl = _label(num, size=16, bold=True, color=YELLOW)
            num_lbl.setFixedWidth(26)
            num_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            txt_layout = QVBoxLayout()
            txt_layout.setSpacing(2)
            txt_layout.addWidget(_label(title, size=12, bold=True))
            txt_layout.addWidget(_label(desc,  size=11, color=TEXT_SEC))
            row.addWidget(num_lbl)
            row.addLayout(txt_layout, 1)
            info_layout.addLayout(row)
            if num != "3":
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: rgba(250,204,21,40);")
                info_layout.addWidget(sep)

        mid.addWidget(nav_panel, 2)
        mid.addWidget(info_panel, 3)
        root.addLayout(mid, 1)


# ---------------------------------------------------------------------------
# Screen 2 – Process Videos
# ---------------------------------------------------------------------------

class ProcessVideosScreen(BackgroundMixin, QWidget):

    def __init__(self, navigate, parent=None):
        super().__init__(parent)
        self._navigate  = navigate
        self._process   = None       # QProcess handle
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # ── Header row ──────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        back_btn = PolyButton("← Menu", primary=False, small=True)
        back_btn.clicked.connect(lambda: self._navigate("menu"))
        hdr.addWidget(back_btn)
        hdr.addStretch()
        hdr.addWidget(_label("Process Videos", size=22, bold=True))
        hdr.addStretch()
        hdr.addSpacing(back_btn.sizeHint().width())
        root.addLayout(hdr)

        # ── Main content splitter ────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: rgba(250,204,21,30); }")
        root.addWidget(splitter, 1)

        # ─── Left: path selection & options ─────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet("background: transparent; border: none;")
        left_inner = QWidget()
        left_inner.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 8, 0)

        # Paths panel
        paths_panel = GlassPanel()
        paths_layout = QFormLayout(paths_panel)
        paths_layout.setContentsMargins(20, 16, 20, 16)
        paths_layout.setSpacing(10)
        paths_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        paths_layout.addRow(_section_title("Input & Output"))

        # Input video folder
        vid_row, self._input_le = _path_row(
            "Select Video Folder", "Folder containing .mp4 / .avi videos", pick_dir=True)
        paths_layout.addRow(_label("Video folder:", color=TEXT_SEC), vid_row)

        # Config file
        cfg_row, self._config_le = _path_row(
            "Select Config File", "config.yaml",
            pick_dir=False, pick_file_filter="YAML files (*.yaml *.yml)")
        paths_layout.addRow(_label("Config file:", color=TEXT_SEC), cfg_row)

        # Output directory
        out_row, self._output_le = _path_row(
            "Select Output Folder", "Where to save results", pick_dir=True)
        paths_layout.addRow(_label("Output folder:", color=TEXT_SEC), out_row)

        left_layout.addWidget(paths_panel)

        # Options panel
        opt_panel = GlassPanel()
        opt_layout = QVBoxLayout(opt_panel)
        opt_layout.setContentsMargins(20, 16, 20, 16)
        opt_layout.setSpacing(10)
        opt_layout.addWidget(_section_title("Options"))

        self._override_cb  = QCheckBox("Override existing output directories")
        self._skip_cb      = QCheckBox("Skip videos that already have output")
        for cb in (self._override_cb, self._skip_cb):
            cb.setStyleSheet("color: #F8F8F2; font-size: 12px;")
        opt_layout.addWidget(self._override_cb)
        opt_layout.addWidget(self._skip_cb)

        left_layout.addWidget(opt_panel)
        left_layout.addStretch()

        # Run controls
        ctrl_panel = GlassPanel()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(20, 14, 20, 14)
        ctrl_layout.setSpacing(10)

        self._progress_lbl = _label("Ready.", size=11, color=TEXT_SEC)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate when visible
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background: rgba(20,20,40,200); border: 1px solid rgba(250,204,21,60);
                           border-radius: 4px; height: 8px; text-align: center; }
            QProgressBar::chunk { background: #FACC15; border-radius: 4px; }
        """)

        btn_row = QHBoxLayout()
        self._run_btn  = PolyButton("Run Polytrack", primary=True)
        self._stop_btn = PolyButton("Stop",          primary=False)
        self._stop_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_process)
        self._stop_btn.clicked.connect(self._stop_process)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._stop_btn)

        ctrl_layout.addWidget(self._progress_lbl)
        ctrl_layout.addWidget(self._progress_bar)
        ctrl_layout.addLayout(btn_row)
        left_layout.addWidget(ctrl_panel)

        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # ─── Right: live log output ──────────────────────────────────────────
        log_panel = GlassPanel()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_layout.setSpacing(8)

        log_hdr = QHBoxLayout()
        log_hdr.addWidget(_section_title("Live Output"))
        log_hdr.addStretch()
        clear_btn = PolyButton("Clear", primary=False, small=True)
        clear_btn.setMinimumWidth(70)
        log_hdr.addWidget(clear_btn)
        log_layout.addLayout(log_hdr)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFont(_monospace_font(10))
        self._log_box.setStyleSheet("""
            QTextEdit {
                background-color: rgba(5,5,15,230);
                color: #C8C8D8;
                border: 1px solid rgba(250,204,21,40);
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #FACC15;
                selection-color: #0D0D0D;
            }
        """)
        clear_btn.clicked.connect(self._log_box.clear)
        log_layout.addWidget(self._log_box, 1)

        splitter.addWidget(log_panel)
        splitter.setSizes([380, 580])

    # --- process control ---

    def _validate_inputs(self) -> bool:
        errors = []
        if not self._input_le.text().strip():
            errors.append("• Video folder is required.")
        elif not os.path.isdir(self._input_le.text().strip()):
            errors.append("• Video folder does not exist.")
        if not self._config_le.text().strip():
            errors.append("• Config file is required.")
        elif not os.path.isfile(self._config_le.text().strip()):
            errors.append("• Config file does not exist.")
        if self._output_le.text().strip() and not os.path.isdir(self._output_le.text().strip()):
            try:
                os.makedirs(self._output_le.text().strip(), exist_ok=True)
            except OSError as e:
                errors.append(f"• Could not create output folder: {e}")
        if errors:
            QMessageBox.warning(self, "Missing inputs", "\n".join(errors))
            return False
        return True

    def _start_process(self):
        if not self._validate_inputs():
            return

        self._log_box.clear()
        self._log_append("[Polytrack] Starting…", color=YELLOW.name())

        main_py = str(Path(__file__).parent / "main.py")
        args = ["python", main_py,
                "--config",    self._config_le.text().strip(),
                "--input-dir", self._input_le.text().strip()]
        if self._output_le.text().strip():
            args += ["--output-dir", self._output_le.text().strip()]
        if self._override_cb.isChecked():
            args.append("--override-output")
        if self._skip_cb.isChecked():
            args.append("--skip-existing")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyRead.connect(self._on_data)
        self._process.finished.connect(self._on_finished)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_lbl.setText("Running…")

        self._process.start(args[0], args[1:])

    def _stop_process(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log_append("[Polytrack] Stopping…", color=WARNING.name())
            self._process.terminate()

    def _on_data(self):
        if self._process:
            raw = bytes(self._process.readAll()).decode("utf-8", errors="replace")
            for line in raw.splitlines():
                col = None
                if "ERROR" in line or "error" in line:
                    col = ERROR.name()
                elif "WARNING" in line or "warning" in line:
                    col = WARNING.name()
                elif line.startswith("[") or "Processing" in line:
                    col = YELLOW.name()
                self._log_append(line, color=col)

    def _on_finished(self, exit_code: int, _exit_status):
        if exit_code == 0:
            self._log_append(
                "[Polytrack] Finished successfully.", color=SUCCESS.name())
            self._progress_lbl.setText("Done.")
        else:
            self._log_append(
                f"[Polytrack] Exited with code {exit_code}.", color=ERROR.name())
            self._progress_lbl.setText(f"Exited with code {exit_code}.")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setVisible(False)

    def _log_append(self, text: str, color: str | None = None):
        col = color or TEXT_PRI.name()
        self._log_box.append(f'<span style="color:{col};">{text}</span>')
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Screen 3 – Configure
# ---------------------------------------------------------------------------

class ConfigureScreen(BackgroundMixin, QWidget):
    """
    Allow ecologists to load a YAML config and tweak the most-used settings
    through form widgets.  A YAML preview pane shows the current file content.
    """

    # Flat list of (yaml_path, label, widget_type, kwargs)
    # yaml_path uses dot-notation into the nested config structure.
    _FIELDS = [
        # source
        ("source.compressed_video",  "Compressed video",    "bool",   {}),
        ("source.skip_frames",       "Skip alternate frames","bool",   {}),
        ("source.device",            "Compute device",      "choice",
             {"options": ["auto", "cpu", "cuda:0", "mps"]}),
        # output
        ("output.save",              "Save output video",   "bool",   {}),
        ("output.show",              "Preview during run",  "bool",   {}),
        ("output.log_level",         "Log level",           "choice",
             {"options": ["DEBUG", "INFO", "WARNING", "ERROR"]}),
        # insect tracking
        ("insect_tracking.min_track_length",         "Min track length (frames)", "int",
             {"min": 1, "max": 9999, "step": 1}),
        ("insect_tracking.max_occlusions",           "Max occlusion frames",     "int",
             {"min": 0, "max": 9999, "step": 1}),
        ("insect_tracking.detection_interval",       "Detection interval (frames)","int",
             {"min": 1, "max": 99999, "step": 10}),
        ("insect_tracking.new_track_distance_thresh","New track distance (px)",  "int",
             {"min": 0, "max": 500, "step": 1}),
        # flower tracking
        ("flower_tracking.track",               "Enable flower tracking",  "bool",   {}),
        ("flower_tracking.detection_interval",  "Flower detect interval",  "int",
             {"min": 1, "max": 99999, "step": 100}),
        ("flower_tracking.border_extension",    "Flower border extension", "float",
             {"min": 0.1, "max": 10.0, "step": 0.1, "decimals": 2}),
    ]

    def __init__(self, navigate, parent=None):
        super().__init__(parent)
        self._navigate   = navigate
        self._config_data: dict = {}
        self._widgets: dict[str, QWidget] = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        back_btn = PolyButton("← Menu", primary=False, small=True)
        back_btn.clicked.connect(lambda: self._navigate("menu"))
        hdr.addWidget(back_btn)
        hdr.addStretch()
        hdr.addWidget(_label("Configure", size=22, bold=True))
        hdr.addStretch()
        hdr.addSpacing(back_btn.sizeHint().width())
        root.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: rgba(250,204,21,30); }")
        root.addWidget(splitter, 1)

        # ─── Left: file selector + parameters ───────────────────────────────
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_vbox = QVBoxLayout(left_widget)
        left_vbox.setContentsMargins(0, 0, 8, 0)
        left_vbox.setSpacing(12)

        # Config file picker
        file_panel = GlassPanel()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(20, 14, 20, 14)
        file_layout.setSpacing(8)
        file_layout.addWidget(_section_title("Config File"))

        cfg_row, self._cfg_le = _path_row(
            "Select Config", "config.yaml",
            pick_dir=False, pick_file_filter="YAML files (*.yaml *.yml)")
        file_layout.addLayout(cfg_row)

        load_btn = PolyButton("Load Config", primary=True, small=True)
        save_btn = PolyButton("Save Config", primary=False, small=True)
        template_btn = PolyButton("New Template", primary=False, small=True)
        load_btn.clicked.connect(self._load_config)
        save_btn.clicked.connect(self._save_config)
        template_btn.clicked.connect(self._create_template)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        for b in (load_btn, save_btn, template_btn):
            btn_row.addWidget(b)
        file_layout.addLayout(btn_row)
        left_vbox.addWidget(file_panel)

        # Parameters panel
        param_panel = GlassPanel()
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setStyleSheet("background: transparent; border: none;")
        param_inner = QWidget()
        param_inner.setStyleSheet("background: transparent;")
        param_form  = QFormLayout(param_inner)
        param_form.setContentsMargins(20, 12, 20, 12)
        param_form.setSpacing(9)
        param_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        param_form.addRow(_section_title("Key Parameters"))
        info = _label(
            "Load a config file above to enable editing.",
            size=11, color=TEXT_SEC)
        info.setWordWrap(True)
        param_form.addRow(info)
        self._param_info_lbl = info

        self._param_form = param_form
        self._param_inner = param_inner
        self._widgets_start_row = param_form.rowCount()

        param_scroll.setWidget(param_inner)
        pv = QVBoxLayout(param_panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(param_scroll)
        left_vbox.addWidget(param_panel, 1)
        splitter.addWidget(left_widget)

        # ─── Right: YAML preview ─────────────────────────────────────────────
        right_panel = GlassPanel()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(8)

        rh = QHBoxLayout()
        rh.addWidget(_section_title("YAML Preview"))
        rh.addStretch()
        refresh_btn = PolyButton("Refresh", primary=False, small=True)
        refresh_btn.setMinimumWidth(80)
        refresh_btn.clicked.connect(self._refresh_preview)
        rh.addWidget(refresh_btn)
        right_layout.addLayout(rh)

        self._yaml_preview = QTextEdit()
        self._yaml_preview.setReadOnly(True)
        self._yaml_preview.setFont(_monospace_font(10))
        self._yaml_preview.setStyleSheet("""
            QTextEdit {
                background-color: rgba(5,5,15,230);
                color: #C8C8D8;
                border: 1px solid rgba(250,204,21,40);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self._yaml_preview.setPlaceholderText("Load a config file to see its contents here.")
        right_layout.addWidget(self._yaml_preview, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([480, 480])

    # --- helpers ---

    def _get_nested(self, d: dict, path: str):
        keys = path.split(".")
        cur  = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _set_nested(self, d: dict, path: str, value):
        keys = path.split(".")
        cur  = d
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value

    def _load_config(self):
        path = self._cfg_le.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a config file first.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not found", f"File not found:\n{path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._config_data = yaml.safe_load(fh) or {}
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._rebuild_form()
        self._refresh_preview()
        self._param_info_lbl.setText(f"Editing: {os.path.basename(path)}")

    def _rebuild_form(self):
        # Remove old widget rows (keep the first _widgets_start_row rows)
        while self._param_form.rowCount() > self._widgets_start_row:
            self._param_form.removeRow(self._widgets_start_row)
        self._widgets.clear()

        last_section = ""
        for yaml_path, label_text, wtype, kwargs in self._FIELDS:
            section = yaml_path.split(".")[0]
            if section != last_section:
                last_section = section
                sec_label = _label(section.replace("_", " ").title(),
                                   size=11, bold=True, color=YELLOW)
                sec_label.setContentsMargins(0, 8, 0, 0)
                self._param_form.addRow(sec_label)

            value = self._get_nested(self._config_data, yaml_path)

            if wtype == "bool":
                w = QCheckBox()
                w.setChecked(bool(value) if value is not None else False)
                w.setStyleSheet("color: #F8F8F2;")
            elif wtype == "choice":
                w = QComboBox()
                w.setStyleSheet("""
                    QComboBox { background: rgba(20,20,40,220); color: #F8F8F2;
                                border: 1px solid rgba(250,204,21,80); border-radius:4px; padding: 3px 6px; }
                    QComboBox QAbstractItemView { background: #1A1A2E; color: #F8F8F2; }
                """)
                for opt in kwargs.get("options", []):
                    w.addItem(opt)
                idx = w.findText(str(value) if value is not None else "")
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif wtype == "int":
                w = QSpinBox()
                _style_input(w)
                w.setRange(kwargs.get("min", 0), kwargs.get("max", 99999))
                w.setSingleStep(kwargs.get("step", 1))
                w.setValue(int(value) if value is not None else 0)
            elif wtype == "float":
                w = QDoubleSpinBox()
                _style_input(w)
                w.setRange(kwargs.get("min", 0.0), kwargs.get("max", 100.0))
                w.setSingleStep(kwargs.get("step", 0.1))
                w.setDecimals(kwargs.get("decimals", 2))
                w.setValue(float(value) if value is not None else 0.0)
            else:
                w = QLineEdit(str(value) if value is not None else "")
                _style_input(w)

            self._widgets[yaml_path] = w
            lbl = _label(label_text + ":", size=11, color=TEXT_SEC)
            self._param_form.addRow(lbl, w)

    def _save_config(self):
        path = self._cfg_le.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a config file first.")
            return
        if not self._config_data:
            QMessageBox.warning(self, "No data", "Load a config before saving.")
            return

        # Collect widget values back into config_data
        for yaml_path, w in self._widgets.items():
            if isinstance(w, QCheckBox):
                val = w.isChecked()
            elif isinstance(w, QComboBox):
                val = w.currentText()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                val = w.value()
            else:
                val = w.text()
            self._set_nested(self._config_data, yaml_path, val)

        try:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(self._config_data, fh, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        self._refresh_preview()
        QMessageBox.information(self, "Saved", f"Config saved to:\n{path}")

    def _create_template(self):
        template_dir = Path(__file__).parent.parent / "config" / "custom"
        template_dir.mkdir(parents=True, exist_ok=True)

        example = next(template_dir.glob("*.yaml"), None)
        if example is None:
            example = Path(__file__).parent.parent / "config" / "config.yaml"

        dest, _ = QFileDialog.getSaveFileName(
            self, "Save new config template", str(template_dir),
            "YAML files (*.yaml *.yml)")
        if not dest:
            return
        if example and example.exists():
            shutil.copy(example, dest)
        else:
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write("# Polytrack config – fill in the sections below\n")
                fh.write("directories:\n  source: ''\n  output: ''\n")
        self._cfg_le.setText(dest)
        self._load_config()

    def _refresh_preview(self):
        path = self._cfg_le.text().strip()
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._yaml_preview.setPlainText(fh.read())
            except Exception as exc:
                self._yaml_preview.setPlainText(f"# Could not read file:\n# {exc}")
        elif self._config_data:
            self._yaml_preview.setPlainText(
                yaml.dump(self._config_data, default_flow_style=False, allow_unicode=True))


# ---------------------------------------------------------------------------
# Screen 4 – View Results
# ---------------------------------------------------------------------------

class ResultsScreen(BackgroundMixin, QWidget):

    def __init__(self, navigate, parent=None):
        super().__init__(parent)
        self._navigate = navigate
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        back_btn = PolyButton("← Menu", primary=False, small=True)
        back_btn.clicked.connect(lambda: self._navigate("menu"))
        hdr.addWidget(back_btn)
        hdr.addStretch()
        hdr.addWidget(_label("View Results", size=22, bold=True))
        hdr.addStretch()
        hdr.addSpacing(back_btn.sizeHint().width())
        root.addLayout(hdr)

        # ── Content ─────────────────────────────────────────────────────────
        main_panel = GlassPanel()
        ml = QVBoxLayout(main_panel)
        ml.setContentsMargins(24, 20, 24, 20)
        ml.setSpacing(12)

        # Folder picker
        ml.addWidget(_section_title("Output Folder"))
        folder_row, self._folder_le = _path_row(
            "Select Output Folder", "Folder containing Polytrack output", pick_dir=True)
        scan_btn = PolyButton("Scan", primary=True, small=True)
        scan_btn.setMinimumWidth(80)
        scan_btn.clicked.connect(self._scan_folder)
        # Append scan button into the row
        folder_row.addWidget(scan_btn)
        ml.addLayout(folder_row)

        # Stats row
        self._stats_lbl = _label("No folder loaded.", size=11, color=TEXT_SEC)
        ml.addWidget(self._stats_lbl)

        # Splitter: file list + detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: rgba(250,204,21,30); }")
        ml.addWidget(splitter, 1)

        # File list
        list_panel = GlassPanel()
        lv = QVBoxLayout(list_panel)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(_label("Filter:", size=11, color=TEXT_SEC))
        self._filter_cb = QComboBox()
        for opt in ["All files", "Videos (.mp4, .avi)", "Data (.csv)", "Images (.png, .jpg)", "Logs (.log, .txt)"]:
            self._filter_cb.addItem(opt)
        self._filter_cb.setStyleSheet("""
            QComboBox { background: rgba(20,20,40,220); color: #F8F8F2;
                        border: 1px solid rgba(250,204,21,80); border-radius:4px; padding:3px 6px; }
            QComboBox QAbstractItemView { background: #1A1A2E; color: #F8F8F2; }
        """)
        self._filter_cb.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_cb, 1)
        lv.addLayout(filter_row)

        self._file_list = QListWidget()
        self._file_list.setStyleSheet("""
            QListWidget { background: rgba(5,5,15,200); color: #C8C8D8;
                          border: 1px solid rgba(250,204,21,40); border-radius:8px; padding:4px; }
            QListWidget::item { padding: 6px 8px; border-radius: 4px; }
            QListWidget::item:selected { background: rgba(250,204,21,60); color: #F8F8F2; }
            QListWidget::item:hover { background: rgba(250,204,21,25); }
        """)
        self._file_list.setFont(_monospace_font(10))
        lv.addWidget(self._file_list, 1)

        splitter.addWidget(list_panel)

        # Detail / action panel
        detail_panel = GlassPanel()
        dv = QVBoxLayout(detail_panel)
        dv.setContentsMargins(16, 14, 16, 14)
        dv.setSpacing(10)
        dv.addWidget(_section_title("File Actions"))

        self._detail_lbl = _label(
            "Select a file on the left to see options.", size=11, color=TEXT_SEC)
        self._detail_lbl.setWordWrap(True)
        dv.addWidget(self._detail_lbl)

        self._open_btn  = PolyButton("Open File", primary=True)
        self._finder_btn = PolyButton("Show in Finder", primary=False)
        self._open_btn.setEnabled(False)
        self._finder_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_file)
        self._finder_btn.clicked.connect(self._show_in_finder)
        dv.addWidget(self._open_btn)
        dv.addWidget(self._finder_btn)
        dv.addStretch()

        # Summary stats
        dv.addWidget(_section_title("Run Summary"))
        self._summary_box = QTextEdit()
        self._summary_box.setReadOnly(True)
        self._summary_box.setFont(_monospace_font(10))
        self._summary_box.setMaximumHeight(200)
        self._summary_box.setStyleSheet("""
            QTextEdit { background: rgba(5,5,15,200); color: #C8C8D8;
                        border: 1px solid rgba(250,204,21,40); border-radius:8px; padding:8px; }
        """)
        self._summary_box.setPlaceholderText("Scan a folder to see summary statistics.")
        dv.addWidget(self._summary_box)

        splitter.addWidget(detail_panel)
        splitter.setSizes([500, 340])
        root.addWidget(main_panel, 1)

        self._file_list.currentItemChanged.connect(self._on_file_selected)
        self._all_files: list[str] = []

    def _scan_folder(self):
        folder = self._folder_le.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please select a valid output folder.")
            return

        self._all_files = []
        for root_dir, _dirs, files in os.walk(folder):
            for fname in sorted(files):
                self._all_files.append(os.path.join(root_dir, fname))

        self._apply_filter()
        n = len(self._all_files)
        self._stats_lbl.setText(
            f"Found {n} file{'s' if n != 1 else ''} in {folder}")
        self._build_summary(folder)

    def _apply_filter(self):
        text = self._filter_cb.currentText()
        ext_map = {
            "Videos (.mp4, .avi)":      (".mp4", ".avi", ".mov"),
            "Data (.csv)":               (".csv",),
            "Images (.png, .jpg)":       (".png", ".jpg", ".jpeg"),
            "Logs (.log, .txt)":         (".log", ".txt"),
        }
        exts = ext_map.get(text, None)

        self._file_list.clear()
        folder = self._folder_le.text().strip()
        for path in self._all_files:
            if exts and not path.lower().endswith(exts):
                continue
            rel = os.path.relpath(path, folder)
            item = QListWidgetItem(rel)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._file_list.addItem(item)

    def _on_file_selected(self, current, _prev):
        if current is None:
            self._detail_lbl.setText("Select a file on the left to see options.")
            self._open_btn.setEnabled(False)
            self._finder_btn.setEnabled(False)
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        size_kb = os.path.getsize(path) / 1024
        self._detail_lbl.setText(
            f"{os.path.basename(path)}\n{size_kb:,.1f} KB\n{path}")
        self._open_btn.setEnabled(True)
        self._finder_btn.setEnabled(True)

    def _open_file(self):
        item = self._file_list.currentItem()
        if item:
            QDesktopServices.openUrl(
                __import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl.fromLocalFile(
                    item.data(Qt.ItemDataRole.UserRole)))

    def _show_in_finder(self):
        item = self._file_list.currentItem()
        if item:
            path = item.data(Qt.ItemDataRole.UserRole)
            subprocess.run(["open", "-R", path], check=False)

    def _build_summary(self, folder: str):
        csv_files  = [f for f in self._all_files if f.endswith(".csv")]
        vid_files  = [f for f in self._all_files if f.lower().endswith((".mp4", ".avi"))]
        img_files  = [f for f in self._all_files if f.lower().endswith((".png", ".jpg"))]
        total_mb   = sum(os.path.getsize(f) for f in self._all_files) / (1024 * 1024)

        lines = [
            f"Output folder : {os.path.basename(folder)}",
            f"Total size    : {total_mb:,.1f} MB",
            f"CSV data files: {len(csv_files)}",
            f"Video outputs : {len(vid_files)}",
            f"Images        : {len(img_files)}",
        ]
        self._summary_box.setPlainText("\n".join(lines))


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class PolytrackApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polytrack")
        self.resize(1200, 780)
        self.setMinimumSize(900, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._screens: dict[str, QWidget] = {}
        self._add_screen("menu",      MainMenuScreen(self._navigate))
        self._add_screen("process",   ProcessVideosScreen(self._navigate))
        self._add_screen("configure", ConfigureScreen(self._navigate))
        self._add_screen("results",   ResultsScreen(self._navigate))

        self._navigate("menu")

    def _add_screen(self, key: str, screen: QWidget):
        self._screens[key] = screen
        self._stack.addWidget(screen)

    def _navigate(self, key: str):
        if key in self._screens:
            self._stack.setCurrentWidget(self._screens[key])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Polytrack")
    app.setOrganizationName("Polytrack")

    # Global stylesheet for common Qt widgets
    app.setStyleSheet("""
        QWidget {
            font-family: "Helvetica Neue", Arial, sans-serif;
            font-size: 12px;
            color: #F8F8F2;
        }
        QScrollBar:vertical {
            background: rgba(20,20,40,150);
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: rgba(250,204,21,120);
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: rgba(20,20,40,150);
            height: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: rgba(250,204,21,120);
            border-radius: 5px;
            min-width: 30px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        QToolTip {
            background: #1A1A2E;
            color: #F8F8F2;
            border: 1px solid #FACC15;
            border-radius: 4px;
            padding: 4px;
        }
        QMessageBox {
            background: #1A1A2E;
            color: #F8F8F2;
        }
    """)

    window = PolytrackApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
