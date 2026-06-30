"""
polytrack_ui.py
===============
PySide6 graphical interface for Polytrack – insect tracking and pollination
monitoring.  Refactored to a two-stage workflow:

    Stage 1 — Setup       (configuration: rail of 4 sections + form)
    Stage 2 — Processing  (batch dashboard: queue, stats, live preview, console)

The app opens directly on Setup.  A header stepper shows ① Setup → ② Processing.
Theming supports Auto (OS-follow) / Light / Dark.

Run
---
    python polytrack_ui.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import signal
import tempfile
from pathlib import Path

import yaml

from PySide6.QtCore import (
    Qt, QProcess, QTimer, QPointF, QRectF, QSize, QObject, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient, QBrush,
    QPixmap, QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox,
    QScrollArea, QSizePolicy, QFrame, QMessageBox, QProgressBar, QInputDialog,
)

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

_SHARED = {
    "yellow":       QColor("#FACC15"),
    "yellow_hover": QColor("#FFE066"),
    "ink":          QColor("#0D0D0D"),
    "ok":           QColor("#22C55E"),
    "warn":         QColor("#F97316"),
    "err":          QColor("#EF4444"),
    "info":         QColor("#60A5FA"),
}

DARK = {
    **_SHARED,
    "scheme":   "dark",
    "bg0":      QColor("#1B1B3A"),
    "bg1":      QColor("#0F0F1F"),
    "bg2":      QColor("#070710"),
    "glass":    QColor(22, 22, 42, 184),     # rgba(22,22,42,.72)
    "border":   QColor(250, 204, 21, 56),    # rgba(250,204,21,.22)
    "text_pri": QColor("#F4F4F6"),
    "text_sec": QColor("#A6A6BA"),
    "text_dim": QColor("#6E6E86"),
    "field":    QColor(10, 10, 24, 179),     # rgba(10,10,24,.7)
    "grid":     QColor(250, 204, 21, 18),
}

LIGHT = {
    **_SHARED,
    "scheme":   "light",
    "bg0":      QColor("#FFFDF4"),
    "bg1":      QColor("#F3EEDE"),
    "bg2":      QColor("#E7DEC6"),
    "glass":    QColor(255, 253, 247, 240),  # rgba(255,253,247,.94)
    "border":   QColor(202, 159, 10, 107),   # rgba(202,159,10,.42)
    "text_pri": QColor("#1B1A12"),
    "text_sec": QColor("#5A5642"),
    "text_dim": QColor("#8A8470"),
    "field":    QColor(255, 255, 255, 255),
    "grid":     QColor(202, 159, 10, 30),
}

# Active palette – mutated in place so painted widgets pick it up live.
PAL: dict = dict(DARK)


def rgba(c: QColor) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha() / 255:.3f})"


# ---------------------------------------------------------------------------
# Theme manager
# ---------------------------------------------------------------------------

class ThemeManager(QObject):
    """Holds the active mode (auto/light/dark) and re-styles the app on change."""

    changed = Signal()

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self._mode = "auto"
        sh = app.styleHints()
        if hasattr(sh, "colorSchemeChanged"):
            sh.colorSchemeChanged.connect(self._on_os_change)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        self._mode = mode.lower()
        self.apply()

    def _resolved(self) -> str:
        if self._mode in ("light", "dark"):
            return self._mode
        sh = self._app.styleHints()
        if hasattr(sh, "colorScheme"):
            try:
                if sh.colorScheme() == Qt.ColorScheme.Light:
                    return "light"
            except Exception:
                pass
        return "dark"

    def apply(self):
        PAL.clear()
        PAL.update(LIGHT if self._resolved() == "light" else DARK)
        self._app.setStyleSheet(app_stylesheet())
        _restyle_all()
        self.changed.emit()

    def _on_os_change(self, _scheme):
        if self._mode == "auto":
            self.apply()


THEME: ThemeManager | None = None     # set in main()


# Registry of (widget, styler_fn) so stylesheet-based widgets follow the theme.
_styled_registry: list[tuple[QWidget, "callable"]] = []


def register_style(widget: QWidget, fn) -> QWidget:
    fn(widget)
    _styled_registry.append((widget, fn))
    return widget


def _restyle_all():
    dead = []
    for w, fn in _styled_registry:
        try:
            fn(w)
        except RuntimeError:
            dead.append((w, fn))
    for d in dead:
        _styled_registry.remove(d)


def track_theme(widget: QWidget):
    """Repaint a custom-painted widget whenever the theme changes."""
    if THEME is not None:
        THEME.changed.connect(widget.update)


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _mono(size: int = 11, bold: bool = False) -> QFont:
    f = QFont("JetBrains Mono")
    f.setStyleHint(QFont.StyleHint.Monospace)
    if not f.exactMatch():
        f = QFont("Courier New")
        f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    f.setBold(bold)
    return f


def _ui(size: int = 13, bold: bool = False) -> QFont:
    f = QFont("Helvetica Neue")
    f.setPointSize(size)
    f.setBold(bold)
    f.setStyleHint(QFont.StyleHint.SansSerif)
    return f


# ---------------------------------------------------------------------------
# Background mixin – radial gradient + perspective grid (theme-aware)
# ---------------------------------------------------------------------------

class BackgroundMixin:

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        grad = QRadialGradient(w / 2, h / 3, max(w, h) * 0.85)
        grad.setColorAt(0.0, PAL["bg0"])
        grad.setColorAt(0.5, PAL["bg1"])
        grad.setColorAt(1.0, PAL["bg2"])
        p.fillRect(0, 0, w, h, grad)

        p.setPen(QPen(PAL["grid"], 1))
        cx, hy = w / 2, h * 0.42
        n_lines, spread = 22, w * 1.1
        for i in range(n_lines + 1):
            x_bot = -spread / 2 + spread * i / n_lines
            p.drawLine(QPointF(cx, hy), QPointF(x_bot, float(h)))
        n_horiz = 14
        for j in range(1, n_horiz + 1):
            frac = j / n_horiz
            y = hy + (h - hy) * frac
            t = frac ** 1.6
            p.drawLine(QPointF(cx - spread / 2 * t, y), QPointF(cx + spread / 2 * t, y))

        super().paintEvent(event)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GlassPanel – theme-aware card
# ---------------------------------------------------------------------------

class GlassPanel(QFrame):

    def __init__(self, parent=None, radius: int = 12):
        super().__init__(parent)
        self._radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent; border: none;")
        track_theme(self)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        p.setBrush(QBrush(PAL["glass"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, self._radius, self._radius)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(PAL["border"], 1.2))
        p.drawRoundedRect(r.adjusted(0.6, 0.6, -0.6, -0.6), self._radius, self._radius)
        super().paintEvent(event)


# ---------------------------------------------------------------------------
# PolyButton – slanted parallelogram push-button (primary / ghost / danger)
# ---------------------------------------------------------------------------

class PolyButton(QPushButton):

    SKEW = 14

    def __init__(self, text: str = "", variant: str = "primary",
                 small: bool = False, parent=None):
        super().__init__(text, parent)
        self.variant = variant
        self.small = small
        self._hovered = False
        self._pressed = False
        h = 36 if small else 46
        self.setMinimumHeight(h)
        self.setMinimumWidth(108 if small else 150)
        self.setFont(_ui(11 if small else 12, bold=True))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        track_theme(self)

    def enterEvent(self, e):  # noqa: N802
        self._hovered = True; self.update(); super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._hovered = self._pressed = False; self.update(); super().leaveEvent(e)

    def mousePressEvent(self, e):  # noqa: N802
        self._pressed = True; self.update(); super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._pressed = False; self.update(); super().mouseReleaseEvent(e)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, sk = self.width(), self.height(), self.SKEW
        path = QPainterPath()
        path.moveTo(sk, 0); path.lineTo(w, 0); path.lineTo(w - sk, h); path.lineTo(0, h)
        path.closeSubpath()

        enabled = self.isEnabled()
        if self.variant == "primary":
            base = PAL["yellow_hover"] if self._hovered else PAL["yellow"]
            if self._pressed:
                base = base.darker(115)
            border = QColor(0, 0, 0, 60)
            text_col = PAL["ink"]
        elif self.variant == "danger":
            base = QColor(PAL["err"]); base.setAlpha(40 if not self._hovered else 70)
            border = PAL["err"]
            text_col = PAL["err"]
        else:  # ghost
            base = QColor(PAL["yellow"]); base.setAlpha(30 if self._hovered else 0)
            border = PAL["yellow"]
            text_col = PAL["yellow"]

        if not enabled:
            base.setAlpha(min(base.alpha(), 40))
            text_col = PAL["text_dim"]
            border = PAL["border"]

        p.setBrush(QBrush(base))
        p.setPen(QPen(border, 1.4))
        p.drawPath(path)
        p.setPen(QPen(text_col))
        p.setFont(self.font())
        p.drawText(path.boundingRect().adjusted(sk / 2, 0, -sk / 2, 0),
                   Qt.AlignmentFlag.AlignCenter, self.text())

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) + self.SKEW * 2 + 36
        h = 36 if self.small else 46
        return QSize(max(text_w, 108 if self.small else 150), h)


# ---------------------------------------------------------------------------
# ToggleSwitch (bool)
# ---------------------------------------------------------------------------

class ToggleSwitch(QWidget):

    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(46, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        track_theme(self)

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, v: bool):  # noqa: N802
        v = bool(v)
        if v != self._checked:
            self._checked = v
            self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        if self._checked:
            p.setBrush(QBrush(PAL["yellow"]))
            p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setBrush(QBrush(PAL["field"]))
            p.setPen(QPen(PAL["border"], 1.2))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = self.height() - 8
        x = self.width() - d - 4 if self._checked else 4
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(PAL["ink"] if self._checked else PAL["text_sec"]))
        p.drawEllipse(QRectF(x, 4, d, d))


# ---------------------------------------------------------------------------
# Segmented control (choice)
# ---------------------------------------------------------------------------

class Segmented(QWidget):

    changed = Signal(str)

    def __init__(self, options: list[str], current: str | None = None, parent=None):
        super().__init__(parent)
        self._options = list(options)
        self._index = 0
        if current in self._options:
            self._index = self._options.index(current)
        self.setFont(_ui(11, bold=True))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        track_theme(self)

    def current(self) -> str:
        return self._options[self._index]

    def set_current(self, value: str):
        if value in self._options:
            self._index = self._options.index(value)
            self.update()

    def _seg_width(self) -> float:
        return self.width() / len(self._options)

    def mouseReleaseEvent(self, e):  # noqa: N802
        idx = int(e.position().x() // self._seg_width())
        idx = max(0, min(len(self._options) - 1, idx))
        if idx != self._index:
            self._index = idx
            self.update()
            self.changed.emit(self.current())
        super().mouseReleaseEvent(e)

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        # segments are drawn equal-width, so size to the widest option × count
        widest = max(fm.horizontalAdvance(o) for o in self._options)
        w = (widest + 24) * len(self._options)
        return QSize(max(160, w), 30)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        p.setBrush(QBrush(PAL["field"]))
        p.setPen(QPen(PAL["border"], 1))
        p.drawRoundedRect(r, 7, 7)
        sw = self._seg_width()
        for i, opt in enumerate(self._options):
            seg = QRectF(i * sw, 0, sw, self.height())
            if i == self._index:
                p.setBrush(QBrush(PAL["yellow"]))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(seg.adjusted(2, 2, -2, -2), 5, 5)
                p.setPen(QPen(PAL["ink"]))
            else:
                p.setPen(QPen(PAL["text_sec"]))
            p.setFont(self.font())
            p.drawText(seg, Qt.AlignmentFlag.AlignCenter, opt)


# ---------------------------------------------------------------------------
# Stepper (int / float)  — [-] value [+]  + unit
# ---------------------------------------------------------------------------

class Stepper(QWidget):

    def __init__(self, value=0, minimum=0, maximum=10 ** 9, step=1,
                 is_float: bool = False, decimals: int = 2, unit: str = "", parent=None):
        super().__init__(parent)
        self._float = is_float
        self._dec = decimals
        self._min = minimum
        self._max = maximum
        self._step = step
        self._value = value

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addStretch()

        self._minus = self._mk_btn("−")
        self._edit = QLineEdit()
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit.setFixedWidth(76)
        self._edit.setFixedHeight(30)
        self._edit.setFont(_mono(11))
        register_style(self._edit, _style_field)
        self._edit.editingFinished.connect(self._on_edit)
        self._plus = self._mk_btn("+")

        lay.addWidget(self._minus)
        lay.addWidget(self._edit)
        lay.addWidget(self._plus)
        if unit:
            u = QLabel(unit)
            u.setFont(_ui(10))
            register_style(u, lambda w: w.setStyleSheet(f"color:{PAL['text_dim'].name()};"))
            u.setFixedWidth(46)
            lay.addWidget(u)

        self._minus.clicked.connect(lambda: self._bump(-self._step))
        self._plus.clicked.connect(lambda: self._bump(self._step))
        self._render()

    def _mk_btn(self, txt) -> QPushButton:
        b = QPushButton(txt)
        b.setFixedSize(30, 30)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFont(_ui(14, bold=True))
        register_style(b, _style_stepbtn)
        return b

    def _bump(self, delta):
        self.set_value(self._value + delta)

    def _on_edit(self):
        txt = self._edit.text().strip()
        try:
            self.set_value(float(txt) if self._float else int(float(txt)))
        except ValueError:
            self._render()

    def _clamp(self, v):
        v = max(self._min, min(self._max, v))
        return round(v, self._dec) if self._float else int(round(v))

    def value(self):
        return self._value

    def set_value(self, v):
        try:
            v = float(v) if self._float else int(round(float(v)))
        except (ValueError, TypeError):
            v = self._min
        self._value = self._clamp(v)
        self._render()

    def _render(self):
        if self._float:
            self._edit.setText(f"{self._value:.{self._dec}f}".rstrip("0").rstrip(".") or "0")
        else:
            self._edit.setText(str(self._value))


# ---------------------------------------------------------------------------
# Status pill (rounded 999px)
# ---------------------------------------------------------------------------

class StatusPill(QLabel):

    def __init__(self, text: str = "", tone: str = "ok", parent=None):
        super().__init__(text, parent)
        self._tone = tone
        self.setFont(_ui(10, bold=True))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        track_theme(self)
        register_style(self, self._restyle)

    def set_tone(self, tone: str, text: str | None = None):
        self._tone = tone
        if text is not None:
            self.setText(text)
        self._restyle(self)

    def _restyle(self, _w=None):
        col = PAL.get(self._tone, PAL["ok"])
        bg = QColor(col); bg.setAlpha(38)
        self.setStyleSheet(
            f"background:{rgba(bg)}; color:{col.name()}; border-radius:11px;"
            f"padding:3px 12px;")


# ---------------------------------------------------------------------------
# Stylesheet helpers (registered so they follow the theme)
# ---------------------------------------------------------------------------

def _style_field(w: QWidget):
    w.setStyleSheet(f"""
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background:{rgba(PAL['field'])};
            color:{PAL['text_pri'].name()};
            border:1px solid {rgba(PAL['border'])};
            border-radius:7px; padding:4px 8px;
            selection-background-color:{PAL['yellow'].name()};
            selection-color:{PAL['ink'].name()};
        }}""")


def _style_stepbtn(w: QWidget):
    w.setStyleSheet(f"""
        QPushButton {{
            background:{rgba(PAL['field'])};
            color:{PAL['yellow'].name()};
            border:1px solid {rgba(PAL['border'])};
            border-radius:7px;
        }}
        QPushButton:hover {{ background:{rgba(PAL['yellow'])}; color:{PAL['ink'].name()}; }}""")


def _style_console(w: QWidget):
    w.setStyleSheet(f"""
        QTextEdit {{
            background:{rgba(PAL['field'])};
            color:{PAL['text_sec'].name()};
            border:1px solid {rgba(PAL['border'])};
            border-radius:8px; padding:8px;
            selection-background-color:{PAL['yellow'].name()};
            selection-color:{PAL['ink'].name()};
        }}""")


def app_stylesheet() -> str:
    return f"""
        QWidget {{ font-family:"Helvetica Neue", Arial, sans-serif; font-size:12px;
                   color:{PAL['text_pri'].name()}; }}
        QScrollArea {{ background:transparent; border:none; }}
        QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
        QScrollBar::handle:vertical {{ background:{rgba(PAL['border'])}; border-radius:5px; min-height:30px; }}
        QScrollBar::handle:vertical:hover {{ background:{rgba(PAL['yellow'])}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
        QScrollBar::handle:horizontal {{ background:{rgba(PAL['border'])}; border-radius:5px; min-width:30px; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
        QComboBox {{ background:{rgba(PAL['field'])}; color:{PAL['text_pri'].name()};
                     border:1px solid {rgba(PAL['border'])}; border-radius:7px; padding:4px 10px; }}
        QComboBox QAbstractItemView {{ background:{PAL['bg1'].name()}; color:{PAL['text_pri'].name()};
                     selection-background-color:{PAL['yellow'].name()}; selection-color:{PAL['ink'].name()}; }}
        QToolTip {{ background:{PAL['bg1'].name()}; color:{PAL['text_pri'].name()};
                    border:1px solid {PAL['yellow'].name()}; border-radius:4px; padding:4px; }}
        QMessageBox, QInputDialog {{ background:{PAL['bg1'].name()}; color:{PAL['text_pri'].name()}; }}
    """


def _label(text: str, size: int = 12, bold: bool = False, tone: str = "text_pri") -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(_ui(size, bold))
    register_style(lbl, lambda w: w.setStyleSheet(f"color:{PAL[tone].name()}; background:transparent;"))
    return lbl


def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    f = _ui(10, bold=True)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
    lbl.setFont(f)
    register_style(lbl, lambda w: w.setStyleSheet(f"color:{PAL['text_dim'].name()}; background:transparent;"))
    return lbl


# ---------------------------------------------------------------------------
# Nested config get / set (supports list indices)
# ---------------------------------------------------------------------------

def get_path(d, path):
    cur = d
    for k in path:
        if isinstance(k, int):
            if isinstance(cur, list) and 0 <= k < len(cur):
                cur = cur[k]
            else:
                return None
        else:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
    return cur


def set_path(d, path, value):
    cur = d
    for i, k in enumerate(path[:-1]):
        nxt = path[i + 1]
        if isinstance(k, int):
            while len(cur) <= k:
                cur.append([] if isinstance(nxt, int) else {})
            cur = cur[k]
        else:
            if not isinstance(cur.get(k), (dict, list)):
                cur[k] = [] if isinstance(nxt, int) else {}
            cur = cur[k]
    last = path[-1]
    if isinstance(last, int):
        while len(cur) <= last:
            cur.append(None)
        cur[last] = value
    else:
        cur[last] = value


# ---------------------------------------------------------------------------
# Logo lockup
# ---------------------------------------------------------------------------

class LogoLockup(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        mark = QLabel("N")
        mf = _ui(20, bold=True)
        mark.setFont(mf)
        register_style(mark, lambda w: w.setStyleSheet(
            f"color:{PAL['yellow'].name()}; border:2px solid {PAL['yellow'].name()};"
            f"border-radius:8px; padding:1px 9px; background:transparent;"))
        lay.addWidget(mark)

        txt = QVBoxLayout(); txt.setSpacing(0)
        self._title = QLabel()
        self._title.setTextFormat(Qt.TextFormat.RichText)
        tf = _ui(17, bold=True)
        tf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        self._title.setFont(tf)
        self._sub = QLabel("v5.0")
        self._sub.setFont(_ui(9, bold=True))
        register_style(self, self._restyle)
        txt.addWidget(self._title)
        txt.addWidget(self._sub)
        lay.addLayout(txt)

    def _restyle(self, _w=None):
        self._title.setText(
            f"<span style='color:{PAL['text_pri'].name()}'>POLY</span>"
            f"<span style='color:{PAL['yellow'].name()}'>TRACK</span>")
        self._sub.setStyleSheet(f"color:{PAL['text_dim'].name()}; background:transparent;")


# ---------------------------------------------------------------------------
# Header stepper chip
# ---------------------------------------------------------------------------

class StepChip(QPushButton):

    def __init__(self, number: int, text: str, parent=None):
        super().__init__(parent)
        self.number = number
        self._text = text
        self._active = False
        self._done = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(_ui(11, bold=True))
        self.setFixedHeight(34)
        self.setStyleSheet("border:none; background:transparent;")
        track_theme(self)

    def set_state(self, active: bool, done: bool):
        self._active, self._done = active, done
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._text) + 64, 34)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        if self._active:
            p.setBrush(QBrush(PAL["yellow"]))
            p.setPen(Qt.PenStyle.NoPen)
        else:
            bg = QColor(PAL["glass"])
            p.setBrush(QBrush(bg))
            p.setPen(QPen(PAL["border"], 1))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 17, 17)

        badge_col = PAL["ink"] if self._active else PAL["text_sec"]
        cr = QRectF(8, (self.height() - 20) / 2, 20, 20)
        if self._active:
            p.setBrush(QBrush(QColor(PAL["ink"])))
        else:
            p.setBrush(QBrush(PAL["yellow"] if self._done else QColor(PAL["border"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cr)
        p.setPen(QPen(PAL["yellow"] if self._active else (PAL["ink"] if self._done else PAL["text_dim"])))
        p.setFont(_ui(10, bold=True))
        p.drawText(cr, Qt.AlignmentFlag.AlignCenter, "✓" if self._done else str(self.number))

        p.setPen(QPen(PAL["ink"] if self._active else PAL["text_sec"]))
        p.setFont(self.font())
        p.drawText(r.adjusted(34, 0, -10, 0),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)


class HeaderBar(QWidget):

    def __init__(self, on_step, on_theme, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        track_theme(self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(16)

        lay.addWidget(LogoLockup())
        lay.addSpacing(12)

        self.step1 = StepChip(1, "Setup")
        self.step2 = StepChip(2, "Processing")
        self.step1.clicked.connect(lambda: on_step(0))
        self.step2.clicked.connect(lambda: on_step(1))
        arrow = QLabel("→"); arrow.setFont(_ui(14, bold=True))
        register_style(arrow, lambda w: w.setStyleSheet(f"color:{PAL['text_dim'].name()};"))
        lay.addWidget(self.step1)
        lay.addWidget(arrow)
        lay.addWidget(self.step2)

        lay.addStretch()

        self.run_pill = StatusPill("● 4 videos detected", tone="text_dim")
        self.run_pill.set_tone("text_dim", "● 4 videos detected")
        lay.addWidget(self.run_pill)

        self.theme_seg = Segmented(["Auto", "Light", "Dark"], "Auto")
        self.theme_seg.setFixedWidth(180)
        self.theme_seg.changed.connect(lambda v: on_theme(v.lower()))
        lay.addWidget(self.theme_seg)

        self.set_stage(0)

    def set_stage(self, idx: int):
        self.step1.set_state(active=(idx == 0), done=(idx > 0))
        self.step2.set_state(active=(idx == 1), done=False)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), PAL["bg1"])
        p.setPen(QPen(PAL["border"], 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


# ---------------------------------------------------------------------------
# Form row helper
# ---------------------------------------------------------------------------

def form_row(title: str, desc: str, control: QWidget) -> QWidget:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 8, 0, 8)
    h.setSpacing(16)
    left = QVBoxLayout(); left.setSpacing(2)
    left.addWidget(_label(title, size=12, bold=True))
    if desc:
        d = _label(desc, size=10, tone="text_dim")
        d.setWordWrap(True)
        left.addWidget(d)
    h.addLayout(left, 1)
    control.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
    h.addWidget(control, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return row


class PathPicker(QWidget):

    def __init__(self, pick_dir: bool = True, file_filter: str = "", parent=None):
        super().__init__(parent)
        self._pick_dir = pick_dir
        self._filter = file_filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(280)
        self.edit.setFixedHeight(32)
        register_style(self.edit, _style_field)
        btn = PolyButton("Browse", variant="ghost", small=True)
        btn.clicked.connect(self._pick)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _pick(self):
        start = self.edit.text().strip() or str(Path.home())
        if self._pick_dir:
            d = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            d, _ = QFileDialog.getOpenFileName(self, "Select file", start, self._filter)
        if d:
            self.edit.setText(d)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, t):  # noqa: N802
        self.edit.setText(str(t) if t else "")


# ---------------------------------------------------------------------------
# Stage 1 — Setup
# ---------------------------------------------------------------------------

RAIL_SECTIONS = [
    ("A", "Input & Output", "Folders, device, video output"),
    ("B", "Insect Tracking", "Detection, motion model, zones"),
    ("C", "Detection Models", "YOLOv8 + foreground segmentation"),
    ("D", "Flower Tracking", "Pollination targets & shading"),
]


class RailItem(QPushButton):

    def __init__(self, badge: str, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.badge, self.title, self.desc = badge, title, desc
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(58)
        self.setStyleSheet("border:none; background:transparent; text-align:left;")
        track_theme(self)

    def set_active(self, v: bool):
        self._active = v
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        if self._active:
            bg = QColor(PAL["yellow"]); bg.setAlpha(28)
            p.setBrush(QBrush(bg))
            p.setPen(QPen(PAL["border"], 1))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 9, 9)
            p.setBrush(QBrush(PAL["yellow"]))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(0, 8, 3, self.height() - 16), 1.5, 1.5)

        br = QRectF(14, (self.height() - 26) / 2, 26, 26)
        p.setBrush(QBrush(PAL["yellow"] if self._active else QColor(PAL["border"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(br, 6, 6)
        p.setPen(QPen(PAL["ink"] if self._active else PAL["text_sec"]))
        p.setFont(_ui(11, bold=True))
        p.drawText(br, Qt.AlignmentFlag.AlignCenter, self.badge)

        p.setPen(QPen(PAL["text_pri"]))
        p.setFont(_ui(12, bold=True))
        p.drawText(QRectF(52, 8, self.width() - 60, 20),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.title)
        p.setPen(QPen(PAL["text_dim"]))
        p.setFont(_ui(9))
        p.drawText(QRectF(52, 30, self.width() - 60, 18),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.desc)


class SetupScreen(BackgroundMixin, QWidget):

    def __init__(self, on_continue, parent=None):
        super().__init__(parent)
        self._on_continue = on_continue
        self._config: dict = {}
        self._config_path: str | None = None
        self._bindings: list[dict] = []
        self._zone_label: QLabel | None = None
        self._tags_label: QLabel | None = None
        self._build()
        self._autoload()

    # ---- construction ----
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 8, 24, 16)
        root.setSpacing(18)

        # Rail
        rail_panel = GlassPanel()
        rail_panel.setFixedWidth(228)
        rl = QVBoxLayout(rail_panel)
        rl.setContentsMargins(14, 18, 14, 18)
        rl.setSpacing(6)
        rl.addWidget(_eyebrow("Configuration"))
        rl.addSpacing(6)
        self._rail_items: list[RailItem] = []
        for i, (badge, title, desc) in enumerate(RAIL_SECTIONS):
            item = RailItem(badge, title, desc)
            item.clicked.connect(lambda _=False, idx=i: self._select_section(idx))
            self._rail_items.append(item)
            rl.addWidget(item)
        rl.addStretch()
        self._runtime_lbl = _label("—", size=11, bold=True, tone="yellow")
        rl.addWidget(_eyebrow("Estimated runtime"))
        rl.addWidget(self._runtime_lbl)
        root.addWidget(rail_panel)

        # Content + action bar
        right = QVBoxLayout()
        right.setSpacing(14)

        content_panel = GlassPanel()
        cv = QVBoxLayout(content_panel)
        cv.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        cv.addWidget(self._stack)
        for builder in (self._sec_io, self._sec_insect, self._sec_models, self._sec_flower):
            self._stack.addWidget(self._scroll(builder()))
        right.addWidget(content_panel, 1)

        # Action bar
        bar = GlassPanel()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(18, 10, 18, 10)
        bl.setSpacing(12)
        self._file_lbl = _label("No config loaded", size=11, tone="text_sec")
        bl.addWidget(_eyebrow("Config"))
        bl.addWidget(self._file_lbl)
        self._valid_pill = StatusPill("checking…", tone="warn")
        bl.addWidget(self._valid_pill)
        bl.addStretch()
        new_btn = PolyButton("New template", variant="ghost", small=True)
        save_btn = PolyButton("Save config", variant="ghost", small=True)
        cont_btn = PolyButton("Continue to Processing  →", variant="primary")
        new_btn.clicked.connect(self._new_template)
        save_btn.clicked.connect(lambda: self._save_config())
        cont_btn.clicked.connect(self._continue)
        bl.addWidget(new_btn)
        bl.addWidget(save_btn)
        bl.addWidget(cont_btn)
        right.addWidget(bar)

        root.addLayout(right, 1)
        self._select_section(0)

    def _scroll(self, inner: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        return sc

    def _section_widget(self, eyebrow: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(28, 22, 28, 22)
        v.setSpacing(2)
        v.addWidget(_label(eyebrow, size=17, bold=True))
        v.addWidget(_label(subtitle, size=11, tone="text_dim"))
        v.addSpacing(10)
        return w, v

    def _group(self, layout: QVBoxLayout, title: str):
        layout.addSpacing(8)
        layout.addWidget(_eyebrow(title))

    def _hsep(self, layout: QVBoxLayout):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        register_style(line, lambda w: w.setStyleSheet(f"background:{rgba(PAL['border'])}; border:none;"))
        layout.addWidget(line)

    # ---- binding registration ----
    def _bind(self, control, path, kind, opts=None):
        self._bindings.append({"control": control, "path": path, "kind": kind, "opts": opts or {}})
        return control

    def _add(self, layout, title, desc, control, path, kind, opts=None):
        self._bind(control, path, kind, opts)
        layout.addWidget(form_row(title, desc, control))

    # ---- Section A : Input & Output ----
    def _sec_io(self) -> QWidget:
        w, v = self._section_widget("Input & Output", "Folders, device, video output")

        self._group(v, "Folders")
        src = PathPicker(pick_dir=True)
        self._add(v, "Source folder", "Directory containing input videos", src,
                  ["directories", "source"], "path")
        out = PathPicker(pick_dir=True)
        self._add(v, "Output folder", "Where Polytrack writes results", out,
                  ["directories", "output"], "path")
        # Run option (not part of the YAML config) — controls the CLI flag.
        self._skip_existing = ToggleSwitch()
        v.addWidget(form_row(
            "Skip existing output",
            "Skip videos that already have an output folder (otherwise overwrite)",
            self._skip_existing))

        self._group(v, "Source")
        self._add(v, "Compressed video", "Input video is compressed", ToggleSwitch(),
                  ["source", "compressed_video"], "bool")
        self._add(v, "Skip frames", "Process only alternate frames", ToggleSwitch(),
                  ["source", "skip_frames"], "bool")
        self._add(v, "Compute device", "Inference device",
                  Segmented(["auto", "cpu", "cuda:0", "mps"]),
                  ["source", "device"], "seg")

        self._group(v, "Output video")
        res = QWidget(); rh = QHBoxLayout(res); rh.setContentsMargins(0, 0, 0, 0); rh.setSpacing(8)
        sw = Stepper(1920, 16, 16000, 2, unit="px")
        sh = Stepper(1080, 16, 16000, 2, unit="px")
        rh.addWidget(sw); rh.addWidget(_label("×", size=13, tone="text_dim")); rh.addWidget(sh)
        self._bind(sw, ["output", "resolution", 0], "num_index")
        self._bind(sh, ["output", "resolution", 1], "num_index")
        v.addWidget(form_row("Resolution", "Output width × height", res))
        self._add(v, "Codec", "FourCC codec for output video",
                  Segmented(["mp4v", "avc1", "XVID", "MJPG"]), ["output", "codec"], "seg")
        self._add(v, "Save output video", "Write annotated video to disk", ToggleSwitch(),
                  ["output", "save"], "bool")
        self._add(v, "Show preview", "Display frames during processing", ToggleSwitch(),
                  ["output", "show"], "bool")
        self._add(v, "Save insect snapshots", "Per-insect crops for verification", ToggleSwitch(),
                  ["output", "save_insect_snapshots"], "bool")
        self._add(v, "Log level", "Console / file logging verbosity",
                  Segmented(["DEBUG", "INFO", "WARNING", "ERROR"]), ["output", "log_level"], "seg")
        v.addStretch()
        return w

    # ---- Section B : Insect Tracking ----
    def _sec_insect(self) -> QWidget:
        w, v = self._section_widget("Insect Tracking", "Detection, motion model, zones")

        self._group(v, "Target classes")
        tags_row = QWidget(); tr = QHBoxLayout(tags_row); tr.setContentsMargins(0, 0, 0, 0); tr.setSpacing(8)
        self._tags_label = StatusPill("bee · class 0", tone="yellow")
        edit_tags = PolyButton("Edit", variant="ghost", small=True)
        edit_tags.clicked.connect(self._edit_tags)
        tr.addWidget(self._tags_label); tr.addWidget(edit_tags)
        v.addWidget(form_row("Tracked labels", "Insect classes from the detection model", tags_row))

        self._group(v, "Algorithms")
        self._add(v, "Motion prediction", "Model used to predict the next position",
                  Segmented(["ConstantVelocity", "Kalman", "EKF"]),
                  ["insect_tracking", "prediction_method"], "seg_map_list",
                  {"map": {"ConstantVelocity": "ConstantVelocity",
                           "Kalman": "KalmanFilter", "EKF": "ExtendedKalmanFilter"}})
        self._add(v, "Track assignment", "Detection → track association",
                  Segmented(["Hungarian", "ABP"]),
                  ["insect_tracking", "assignment_method"], "seg_map_list",
                  {"map": {"Hungarian": "HungarianMethod", "ABP": "ABP"}})

        self._group(v, "Detection & tracking")
        self._add(v, "Detection interval", "Frames between full re-detections",
                  Stepper(300, 1, 100000, 10, unit="frames"),
                  ["insect_tracking", "detection_interval"], "num")
        self._add(v, "Min track length", "Discard tracks shorter than this",
                  Stepper(10, 1, 9999, 1, unit="frames"),
                  ["insect_tracking", "min_track_length"], "num")
        self._add(v, "Max occlusions", "Frames an insect may stay hidden",
                  Stepper(60, 0, 9999, 1, unit="frames"),
                  ["insect_tracking", "max_occlusions"], "num")
        self._add(v, "New-track distance", "Suppress new tracks near existing ones",
                  Stepper(5, 0, 5000, 1, unit="px"),
                  ["insect_tracking", "new_track_distance_thresh"], "num")
        blob = QWidget(); bh = QHBoxLayout(blob); bh.setContentsMargins(0, 0, 0, 0); bh.setSpacing(8)
        bmin = Stepper(5, 0, 100000, 1, unit="px")
        bmax = Stepper(8000, 0, 1000000, 100, unit="px")
        bh.addWidget(bmin); bh.addWidget(_label("→", size=12, tone="text_dim")); bh.addWidget(bmax)
        self._bind(bmin, ["insect_tracking", "min_blob_area"], "num")
        self._bind(bmax, ["insect_tracking", "max_blob_area"], "num")
        v.addWidget(form_row("Blob area range", "Min / max foreground blob size", blob))
        self._add(v, "Jump distance", "Max px an insect moves between frames",
                  Stepper(50, 0, 5000, 1, unit="px"),
                  ["insect_tracking", "jump_distance", 0], "num_index")

        self._group(v, "Spatial filtering — include zone")
        self._add(v, "Restrict tracking to a region", "x1, y1 → x2, y2 in source pixels",
                  ToggleSwitch(), ["insect_tracking", "spatial_filtering", "use_include_zone"], "bool")
        self._zone_label = _label("—", size=11, tone="text_dim")
        zrow = QWidget(); zl = QHBoxLayout(zrow); zl.setContentsMargins(0, 0, 0, 0)
        zl.addStretch(); zl.addWidget(self._zone_label)
        v.addWidget(zrow)
        v.addStretch()
        return w

    # ---- Section C : Detection Models ----
    def _sec_models(self) -> QWidget:
        w, v = self._section_widget("Detection Models", "YOLOv8 + foreground segmentation")
        dlp = ["insect_tracking", "detector_properties", "dl_detection"]

        self._group(v, "Primary detector (YOLOv8)")
        self._add(v, "Model file", "Path to the primary .pt weights",
                  PathPicker(pick_dir=False, file_filter="Model (*.pt *.onnx *.engine)"),
                  dlp + ["model"], "path")
        self._add(v, "Detection confidence", "Minimum confidence for valid detections",
                  Stepper(0.2, 0.0, 1.0, 0.05, is_float=True, decimals=3),
                  dlp + ["detection_confidence"], "num_list")
        self._add(v, "Confidence floor", "Inference floor below which detections are dropped",
                  Stepper(0.001, 0.0, 1.0, 0.001, is_float=True, decimals=3),
                  dlp + ["detection_confidence_floor"], "num_list")
        self._add(v, "Image size", "Inference image width",
                  Stepper(1024, 64, 8192, 32, unit="px"),
                  dlp + ["image_size", 0], "num_index")
        self._add(v, "Use FP16", "Half-precision inference on CUDA GPUs",
                  ToggleSwitch(), dlp + ["use_fp16"], "bool")

        self._group(v, "Secondary verification")
        svp = ["insect_tracking", "detector_properties", "secondary_verification"]
        self._add(v, "Enable", "Validate new detections with a 2nd model", ToggleSwitch(),
                  ["insect_tracking", "detectors_secondary_verification"], "detector_toggle")
        self._add(v, "Model file", "Path to the secondary .pt weights",
                  PathPicker(pick_dir=False, file_filter="Model (*.pt *.onnx *.engine)"),
                  svp + ["model"], "path")
        self._add(v, "Detection confidence", "Confidence for the primary class",
                  Stepper(0.5, 0.0, 1.0, 0.05, is_float=True, decimals=3),
                  svp + ["detection_confidence"], "num_list")

        self._group(v, "Foreground segmentation")
        fgp = ["insect_tracking", "detector_properties", "fgbg_detection"]
        self._add(v, "Method", "Background / foreground model",
                  Segmented(["MOG2", "FrameDifference"]), fgp + ["model"], "seg_list")
        self._add(v, "Movement threshold", "Intensity threshold for motion",
                  Stepper(80, 0, 255, 1), fgp + ["movement_threshold"], "num")
        self._add(v, "Dilate kernel size", "Kernel size for blob dilation",
                  Stepper(4, 0, 64, 1, unit="px"), fgp + ["dilate_kernel_size"], "num")
        self._add(v, "Warmup frames", "MOG2 background warmup",
                  Stepper(5, 0, 1000, 1, unit="frames"), fgp + ["warmup_frames"], "num")
        v.addStretch()
        return w

    # ---- Section D : Flower Tracking ----
    def _sec_flower(self) -> QWidget:
        w, v = self._section_widget("Flower Tracking", "Pollination targets & shading")
        fdp = ["flower_tracking", "detector_properties"]

        self._group(v, "Tracking")
        self._add(v, "Track flowers", "Enable flower detection & shading",
                  ToggleSwitch(), ["flower_tracking", "track"], "bool")
        self._add(v, "Model file", "Path to the flower detection model",
                  PathPicker(pick_dir=False, file_filter="Model (*.pt *.onnx *.engine)"),
                  fdp + ["model"], "path")
        self._add(v, "Detection confidence", "Minimum confidence for valid detections",
                  Stepper(0.1, 0.0, 1.0, 0.05, is_float=True, decimals=3),
                  fdp + ["detection_confidence"], "num")
        self._add(v, "Detection interval", "Frames between flower detection updates",
                  Stepper(4000, 1, 1000000, 100, unit="frames"),
                  ["flower_tracking", "detection_interval"], "num")
        self._add(v, "Image size", "Inference image width",
                  Stepper(1024, 64, 8192, 32, unit="px"), fdp + ["image_size", 0], "num_index")

        self._group(v, "Shading")
        self._add(v, "Output shape", "How flower areas are shaded",
                  Segmented(["Circle", "Box"]), ["flower_tracking", "output_shape"], "seg")
        self._add(v, "Border extension", "Scale factor to extend flower boundaries",
                  Stepper(2.5, 0.1, 20.0, 0.1, is_float=True, decimals=2),
                  ["flower_tracking", "border_extension"], "num")
        self._add(v, "Mark insects on flower", "Mark insect positions inside flower radius",
                  ToggleSwitch(), ["flower_tracking", "mark_insects_on_flower"], "bool")
        v.addStretch()
        return w

    # ---- section nav ----
    def _select_section(self, idx: int):
        self._stack.setCurrentIndex(idx)
        for i, item in enumerate(self._rail_items):
            item.set_active(i == idx)

    # ---- config IO ----
    def _autoload(self):
        candidates = []
        cfg_dir = Path(__file__).resolve().parent.parent / "config"
        candidates += sorted((cfg_dir / "custom").glob("*.yaml")) if (cfg_dir / "custom").exists() else []
        if (cfg_dir / "config.yaml").exists():
            candidates.append(cfg_dir / "config.yaml")
        if candidates:
            self._load_config(str(candidates[0]))
        else:
            self._refresh_validity()

    def _load_config(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh) or {}
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._config_path = path
        self._file_lbl.setText(os.path.basename(path))
        self._populate()
        self._refresh_validity()

    def _populate(self):
        cfg = self._config
        for b in self._bindings:
            ctrl, path, kind, opts = b["control"], b["path"], b["kind"], b["opts"]
            val = get_path(cfg, path)
            try:
                if kind == "bool":
                    ctrl.setChecked(bool(val))
                elif kind == "seg":
                    if val is not None:
                        ctrl.set_current(str(val))
                elif kind == "seg_list":
                    cv = val[0] if isinstance(val, list) and val else val
                    if cv is not None:
                        ctrl.set_current(str(cv))
                elif kind == "seg_map_list":
                    cv = val[0] if isinstance(val, list) and val else val
                    inv = {v: k for k, v in opts["map"].items()}
                    ctrl.set_current(inv.get(cv, ctrl.current()))
                elif kind == "num":
                    if val is not None:
                        ctrl.set_value(val)
                elif kind == "num_list":
                    cv = val[0] if isinstance(val, list) and val else val
                    if cv is not None:
                        ctrl.set_value(cv)
                elif kind == "num_index":
                    if val is not None:
                        ctrl.set_value(val)
                elif kind == "path":
                    ctrl.setText(str(val) if val else "")
                elif kind == "detector_toggle":
                    dets = get_path(cfg, ["insect_tracking", "detectors"]) or []
                    ctrl.setChecked("secondary_verification" in dets)
            except Exception:
                pass
        self._refresh_tags()
        self._refresh_zone()

    def _collect(self):
        cfg = self._config
        for b in self._bindings:
            ctrl, path, kind, opts = b["control"], b["path"], b["kind"], b["opts"]
            try:
                if kind == "bool":
                    set_path(cfg, path, bool(ctrl.isChecked()))
                elif kind == "seg":
                    set_path(cfg, path, ctrl.current())
                elif kind == "seg_list":
                    set_path(cfg, path, [ctrl.current()])
                elif kind == "seg_map_list":
                    set_path(cfg, path, [opts["map"][ctrl.current()]])
                elif kind == "num":
                    set_path(cfg, path, ctrl.value())
                elif kind == "num_list":
                    set_path(cfg, path, [ctrl.value()])
                elif kind == "num_index":
                    set_path(cfg, path, ctrl.value())
                elif kind == "path":
                    set_path(cfg, path, ctrl.text().strip())
                elif kind == "detector_toggle":
                    dets = get_path(cfg, ["insect_tracking", "detectors"])
                    if not isinstance(dets, list):
                        dets = []
                    has = "secondary_verification" in dets
                    if ctrl.isChecked() and not has:
                        dets.append("secondary_verification")
                    elif not ctrl.isChecked() and has:
                        dets = [d for d in dets if d != "secondary_verification"]
                    set_path(cfg, ["insect_tracking", "detectors"], dets)
            except Exception:
                pass

    def _refresh_tags(self):
        labels = get_path(self._config, ["insect_tracking", "labels"]) or []
        classes = get_path(self._config, ["insect_tracking", "classes"]) or []
        parts = []
        for i, lab in enumerate(labels):
            cls = classes[i] if i < len(classes) else "?"
            parts.append(f"{lab} · class {cls}")
        if self._tags_label:
            self._tags_label.set_tone("yellow", "  |  ".join(parts) if parts else "none")

    def _refresh_zone(self):
        coord = get_path(self._config, ["insect_tracking", "spatial_filtering", "include_zone_coord"]) or []
        if self._zone_label:
            if len(coord) >= 4:
                self._zone_label.setText(f"{coord[0]}, {coord[1]}  →  {coord[2]}, {coord[3]}")
            else:
                self._zone_label.setText("not set")

    def _edit_tags(self):
        labels = get_path(self._config, ["insect_tracking", "labels"]) or []
        classes = get_path(self._config, ["insect_tracking", "classes"]) or []
        lab_txt, ok = QInputDialog.getText(self, "Tracked labels",
                                           "Comma-separated labels:", text=", ".join(map(str, labels)))
        if not ok:
            return
        cls_txt, ok = QInputDialog.getText(self, "Tracked classes",
                                           "Comma-separated class IDs:", text=", ".join(map(str, classes)))
        if not ok:
            return
        new_labels = [s.strip() for s in lab_txt.split(",") if s.strip()]
        new_classes = []
        for s in cls_txt.split(","):
            s = s.strip()
            if s:
                try:
                    new_classes.append(int(s))
                except ValueError:
                    new_classes.append(s)
        set_path(self._config, ["insect_tracking", "labels"], new_labels)
        set_path(self._config, ["insect_tracking", "classes"], new_classes)
        self._refresh_tags()

    def _refresh_validity(self) -> bool:
        required = ["directories", "source", "output", "insect_tracking", "flower_tracking"]
        ok = bool(self._config) and all(k in self._config for k in required)
        self._valid_pill.set_tone("ok" if ok else "err", "valid" if ok else "invalid")
        return ok

    def _save_config(self, path: str | None = None) -> str | None:
        self._collect()
        if path is None:
            path = self._config_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save config", str(Path.home()), "YAML files (*.yaml *.yml)")
            if not path:
                return None
        try:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(self._config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return None
        self._config_path = path
        self._file_lbl.setText(os.path.basename(path))
        self._refresh_validity()
        return path

    def _new_template(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "config"
        src = next((cfg_dir / "custom").glob("*.yaml"), None) if (cfg_dir / "custom").exists() else None
        if src is None and (cfg_dir / "config.yaml").exists():
            src = cfg_dir / "config.yaml"
        dest, _ = QFileDialog.getSaveFileName(
            self, "New config template", str(cfg_dir / "custom"), "YAML files (*.yaml *.yml)")
        if not dest:
            return
        try:
            if src and src.exists():
                with open(src) as fh:
                    data = yaml.safe_load(fh) or {}
            else:
                data = {"directories": {"source": "", "output": ""}}
            with open(dest, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.critical(self, "Template error", str(exc))
            return
        self._load_config(dest)

    def _continue(self):
        if not self._refresh_validity():
            QMessageBox.warning(self, "Invalid config", "Config is missing required sections.")
            return
        self._collect()
        # write working run config
        fd, run_path = tempfile.mkstemp(suffix=".yaml", prefix="polytrack_run_")
        os.close(fd)
        with open(run_path, "w", encoding="utf-8") as fh:
            yaml.dump(self._config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._on_continue(run_path, self._config, self._skip_existing.isChecked())


# ---------------------------------------------------------------------------
# Stage 2 — Processing
# ---------------------------------------------------------------------------

VIDEO_SUFFIXES = (".avi", ".mp4", ".h264", ".MTS", ".mov")


class StatCard(GlassPanel):

    def __init__(self, value: str, label: str, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(2)
        self.value = QLabel(value)
        self.value.setFont(_mono(20, bold=True))
        register_style(self.value, lambda w: w.setStyleSheet(f"color:{PAL['text_pri'].name()};"))
        v.addWidget(self.value)
        v.addWidget(_eyebrow(label))

    def set_value(self, text: str):
        self.value.setText(text)


class StatusDot(QWidget):
    """Small status dot.  state: done / running / queued."""

    def __init__(self, state="queued", parent=None):
        super().__init__(parent)
        self.state = state
        self._phase = 0.0
        self.setFixedSize(14, 14)
        track_theme(self)

    def set_state(self, state):
        self.state = state
        self.update()

    def set_phase(self, ph):
        self._phase = ph
        if self.state == "running":
            self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = {"done": PAL["ok"], "running": PAL["yellow"], "queued": PAL["text_dim"]}[self.state]
        if self.state == "running":
            halo = QColor(col); halo.setAlpha(int(60 + 50 * self._phase))
            p.setBrush(QBrush(halo)); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(0, 0, 14, 14))
            p.setBrush(QBrush(col)); p.drawEllipse(QRectF(3.5, 3.5, 7, 7))
        else:
            p.setBrush(QBrush(col)); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(3, 3, 8, 8))


def _style_progress(w: QWidget):
    w.setStyleSheet(f"""
        QProgressBar {{ background:{rgba(PAL['field'])}; border:1px solid {rgba(PAL['border'])};
                        border-radius:5px; height:8px; text-align:center; }}
        QProgressBar::chunk {{ background:{PAL['yellow'].name()}; border-radius:5px; }}""")


class JobRow(QWidget):

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.state = "queued"
        self.tracks = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        top = QHBoxLayout(); top.setSpacing(10)
        self.dot = StatusDot("queued")
        top.addWidget(self.dot)
        col = QVBoxLayout(); col.setSpacing(1)
        self.name_lbl = QLabel(name)
        self.name_lbl.setFont(_mono(11, bold=True))
        register_style(self.name_lbl, lambda w: w.setStyleSheet(f"color:{PAL['text_pri'].name()};"))
        self.sub_lbl = QLabel("queued")
        self.sub_lbl.setFont(_ui(9))
        register_style(self.sub_lbl, lambda w: w.setStyleSheet(f"color:{PAL['text_dim'].name()};"))
        col.addWidget(self.name_lbl); col.addWidget(self.sub_lbl)
        top.addLayout(col, 1)
        self.pct_lbl = QLabel("—")
        self.pct_lbl.setFont(_mono(10, bold=True))
        register_style(self.pct_lbl, lambda w: w.setStyleSheet(f"color:{PAL['text_sec'].name()};"))
        top.addWidget(self.pct_lbl)
        v.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        register_style(self.bar, _style_progress)
        v.addWidget(self.bar)
        register_style(self, self._restyle)

    def _restyle(self, _w=None):
        if self.state == "running":
            bg = QColor(PAL["yellow"]); bg.setAlpha(22)
            self.setStyleSheet(f"background:{rgba(bg)}; border-radius:9px;")
        else:
            self.setStyleSheet("background:transparent; border-radius:9px;")

    def set_state(self, state):
        self.state = state
        self.dot.set_state(state)
        self._restyle()

    def set_ratio(self, ratio: float):
        self.bar.setValue(int(max(0.0, min(1.0, ratio)) * 1000))
        self.pct_lbl.setText(f"{ratio * 100:.0f}%")

    def set_sub(self, text: str):
        self.sub_lbl.setText(text)


class ProcessingScreen(BackgroundMixin, QWidget):

    _RE_PCT = re.compile(r"\]\s*([\d.]+)%")
    _RE_ACTIVE = re.compile(r"(\d+)\s+active tracks")
    _RE_SAVED = re.compile(r"(\d+)\s+saved tracks")
    _RE_FLOWERS = re.compile(r"(\d+)\s+flowers")
    _RE_FPS = re.compile(r"Processing FPS:\s*([\d.]+)")
    _RE_FRAMES = re.compile(r"(\d+)/(\d+)\s+frames")
    _RE_COMPLETED = re.compile(r"Completed\s+(\d+)/(\d+)\s+videos")
    _RE_FINISHED = re.compile(r"Tracking finished for\s+(.+?)\s*\|")
    _RE_SKIP = re.compile(r"Skipping\s+(.+?)\s+\(output directory exists\)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._config: dict = {}
        self._config_path: str | None = None
        self._frame_pipe: str | None = None
        self._frame_timer: QTimer | None = None
        self._last_frame_mtime = 0.0
        self._rows: list[JobRow] = []
        self._total = 0
        self._done_count = 0
        self._cur_index = 0
        self._cur_ratio = 0.0
        self._start_time = 0.0
        self._show_preview = False
        self._paused = False
        self._build()

        self._pulse = QTimer(self)
        self._pulse.timeout.connect(self._tick_pulse)
        self._pulse.start(60)
        self._pulse_phase = 0.0
        self._eta_timer = QTimer(self)
        self._eta_timer.timeout.connect(self._tick_eta)
        self._eta_timer.start(1000)

    # ---- construction ----
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 8, 24, 16)
        root.setSpacing(14)

        # stat strip
        strip = QHBoxLayout(); strip.setSpacing(12)
        self.card_job = StatCard("0%", "Whole job")
        self.card_cur = StatCard("0 / 0", "Current video")
        self.card_fps = StatCard("—", "Throughput  fps")
        self.card_tracks = StatCard("0", "Insect tracks")
        self.card_eta = StatCard("—", "Est. remaining")
        for c in (self.card_job, self.card_cur, self.card_fps, self.card_tracks, self.card_eta):
            strip.addWidget(c, 1)
        root.addLayout(strip)

        # whole-job progress
        jobpanel = GlassPanel()
        jl = QVBoxLayout(jobpanel)
        jl.setContentsMargins(18, 12, 18, 12)
        jl.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(_eyebrow("Whole-job progress"))
        head.addStretch()
        self.job_sub = _label("video 0 of 0 · 00:00 elapsed", size=10, tone="text_dim")
        head.addWidget(self.job_sub)
        jl.addLayout(head)
        self.job_bar = QProgressBar()
        self.job_bar.setRange(0, 1000); self.job_bar.setValue(0)
        self.job_bar.setTextVisible(False); self.job_bar.setFixedHeight(10)
        register_style(self.job_bar, _style_progress)
        jl.addWidget(self.job_bar)
        root.addWidget(jobpanel)

        # body
        body = QHBoxLayout(); body.setSpacing(16)

        # left: queue
        left = GlassPanel()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 14, 16, 14)
        ll.setSpacing(8)
        qh = QHBoxLayout()
        qh.addWidget(_eyebrow("Job queue"))
        qh.addStretch()
        qh.addWidget(_label("per-video", size=9, tone="text_dim"))
        ll.addLayout(qh)
        self._queue_scroll = QScrollArea()
        self._queue_scroll.setWidgetResizable(True)
        self._queue_inner = QWidget()
        self._queue_inner.setStyleSheet("background:transparent;")
        self._queue_layout = QVBoxLayout(self._queue_inner)
        self._queue_layout.setContentsMargins(0, 0, 6, 0)
        self._queue_layout.setSpacing(4)
        self._queue_layout.addStretch()
        self._queue_scroll.setWidget(self._queue_inner)
        ll.addWidget(self._queue_scroll, 1)

        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self.pause_btn = PolyButton("❚❚ Pause", variant="ghost", small=True)
        self.stop_btn = PolyButton("■ Stop", variant="danger", small=True)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self.pause_btn)
        ctrl.addWidget(self.stop_btn)
        ctrl.addStretch()
        self.elapsed_lbl = _label("00:00 / —", size=10, tone="text_sec")
        ctrl.addWidget(self.elapsed_lbl)
        ll.addLayout(ctrl)
        body.addWidget(left, 2)

        # right: preview + console
        right = QVBoxLayout(); right.setSpacing(12)
        self._preview = PreviewWidget()
        right.addWidget(self._preview, 3)

        console_panel = GlassPanel()
        cl = QVBoxLayout(console_panel)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(8)
        ch = QHBoxLayout()
        ch.addWidget(_eyebrow("Console · live output"))
        ch.addStretch()
        self.log_chip = StatusPill("log_level INFO", tone="info")
        ch.addWidget(self.log_chip)
        cl.addLayout(ch)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(_mono(10))
        register_style(self.console, _style_console)
        cl.addWidget(self.console, 1)
        right.addWidget(console_panel, 2)
        body.addLayout(right, 3)

        root.addLayout(body, 1)

    def _tick_pulse(self):
        import math
        self._pulse_phase = (self._pulse_phase + 0.08) % 1.0
        val = (math.sin(self._pulse_phase * 2 * math.pi) + 1) / 2
        for r in self._rows:
            if r.state == "running":
                r.dot.set_phase(val)
        if self._show_preview:
            self._preview.set_rec(val > 0.5)

    def _tick_eta(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            elapsed = time.time() - self._start_time
            ratio = self._whole_ratio()
            eta = (elapsed * (1 - ratio) / ratio) if ratio > 0.02 else 0
            self.card_eta.set_value(f"~{_fmt_time(eta)}" if eta else "—")
            self.elapsed_lbl.setText(f"{_fmt_time(elapsed)} / ~{_fmt_time(eta) if eta else '—'}")
            self.job_sub.setText(f"video {max(1, self._cur_index)} of {self._total} · {_fmt_time(elapsed)} elapsed")

    # ---- run lifecycle ----
    def start_run(self, config_path: str, config: dict, skip_existing: bool = False):
        self._config_path = config_path
        self._config = config
        self._show_preview = bool(get_path(config, ["output", "show"]))
        self.console.clear()
        self._build_queue()
        self._done_count = 0
        self._cur_index = 0
        self._cur_ratio = 0.0
        self._paused = False
        self.pause_btn.setText("❚❚ Pause")
        self._start_time = time.time()

        main_py = str(Path(__file__).parent / "main.py")
        args = [main_py, "--config", config_path,
                "--skip-existing" if skip_existing else "--override-output"]

        self._preview.set_enabled(self._show_preview)
        if self._show_preview:
            fd, self._frame_pipe = tempfile.mkstemp(suffix=".jpg", prefix="polytrack_frame_")
            os.close(fd)
            args += ["--frame-pipe", self._frame_pipe]
            self._frame_timer = QTimer(self)
            self._frame_timer.timeout.connect(self._poll_frame)
            self._frame_timer.start(66)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(str(Path(__file__).parent))
        self._process.readyRead.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(
            lambda err: self._log(f"[ERROR] QProcess: {err}", "err"))

        self._log(f"[INFO] Launching python {Path(main_py).name}", "info")
        self._log(f"[INFO] Config: {config_path}", "text_dim")
        self.stop_btn.setEnabled(True)
        if self._total:
            self._set_running(0)
        self._process.start(sys.executable, args)

    def _build_queue(self):
        for r in self._rows:
            r.setParent(None)
            r.deleteLater()
        self._rows.clear()
        source = get_path(self._config, ["directories", "source"]) or ""
        videos = []
        sp = Path(source)
        if sp.is_dir():
            videos = sorted(str(v.name) for v in sp.iterdir() if v.suffix in VIDEO_SUFFIXES)
            if not videos:
                videos = sorted(str(v.name) for v in sp.rglob("*") if v.suffix in VIDEO_SUFFIXES)
        elif sp.is_file():
            videos = [sp.name]
        if not videos:
            videos = ["(no videos found)"]
        self._total = len([v for v in videos if v != "(no videos found)"])
        for name in videos:
            row = JobRow(name)
            self._rows.append(row)
            self._queue_layout.insertWidget(self._queue_layout.count() - 1, row)
        self.card_cur.set_value(f"0 / {self._total}")

    def _row_by_name(self, name: str) -> JobRow | None:
        for r in self._rows:
            if r.name == name or Path(r.name).name == Path(name).name:
                return r
        return None

    def _set_running(self, idx: int):
        self._cur_index = idx + 1
        for i, r in enumerate(self._rows):
            if i < idx:
                if r.state != "done":
                    r.set_state("done"); r.set_ratio(1.0)
            elif i == idx:
                r.set_state("running")
                r.set_sub("tracking · 0 tracks")
            else:
                r.set_state("queued")
        self.card_cur.set_value(f"{self._cur_index} / {self._total}")

    def _on_stdout(self):
        if not self._process:
            return
        raw = bytes(self._process.readAll()).decode("utf-8", errors="replace")
        for line in re.split(r"[\r\n]+", raw):
            line = line.strip()
            if not line:
                continue
            self._parse(line)
            self._log_line(line)

    def _parse(self, line: str):
        msk = self._RE_SKIP.search(line)
        if msk:
            stem = msk.group(1).strip()
            for r in self._rows:
                if Path(r.name).stem == stem and r.state != "done":
                    idx = self._rows.index(r)
                    r.set_state("done"); r.set_ratio(1.0); r.set_sub("skipped · output exists")
                    self._done_count = max(self._done_count, idx + 1)
                    if idx + 1 < len(self._rows):
                        self._set_running(idx + 1)
                    break
            self._cur_ratio = 0.0
            self._update_whole()
            return

        m = self._RE_FINISHED.search(line)
        if m:
            row = self._row_by_name(m.group(1))
            if row:
                idx = self._rows.index(row)
                row.set_state("done"); row.set_ratio(1.0)
                row.set_sub(f"done · {row.tracks} tracks")
                self._done_count = max(self._done_count, idx + 1)
                if idx + 1 < len(self._rows):
                    self._set_running(idx + 1)
            self._cur_ratio = 0.0
            self._update_whole()
            return

        mc = self._RE_COMPLETED.search(line)
        if mc:
            self._done_count = int(mc.group(1))
            self._update_whole()
            return

        # per-frame progress line begins with the video name
        head = re.match(r"^(\S.*?)\s+\[", line)
        if head:
            row = self._row_by_name(head.group(1))
            if row and row.state != "done":
                idx = self._rows.index(row)
                if row.state != "running":
                    self._set_running(idx)
                mp = self._RE_PCT.search(line)
                if mp:
                    self._cur_ratio = float(mp.group(1)) / 100.0
                    row.set_ratio(self._cur_ratio)
                ms = self._RE_SAVED.search(line)
                ma = self._RE_ACTIVE.search(line)
                if ms:
                    row.tracks = int(ms.group(1))
                    self.card_tracks.set_value(str(row.tracks))
                    row.set_sub(f"tracking · {row.tracks} tracks")
                if ma:
                    self.card_tracks.set_value(ma.group(1))
                self._update_whole()

        mf = self._RE_FRAMES.search(line)
        if mf and self._show_preview:
            self._preview.set_frame_info(int(mf.group(1)), int(mf.group(2)))
        mfps = self._RE_FPS.search(line)
        if mfps:
            fps = float(mfps.group(1))
            self.card_fps.set_value(f"{fps:.1f}")
            self._preview.set_fps(fps)

    def _whole_ratio(self) -> float:
        if not self._total:
            return 0.0
        return min(1.0, (self._done_count + self._cur_ratio) / self._total)

    def _update_whole(self):
        r = self._whole_ratio()
        self.job_bar.setValue(int(r * 1000))
        self.card_job.set_value(f"{r * 100:.0f}%")
        if self._rows and self._cur_index:
            cur = self._rows[min(self._cur_index - 1, len(self._rows) - 1)]
            self._preview.set_filename(cur.name)

    def _log_line(self, line: str):
        if re.search(r"\bERROR\b|error", line):
            tone = "err"
        elif re.search(r"\bWARN", line) or "warning" in line.lower():
            tone = "warn"
        elif re.search(r"\bOK\b|finished|complete", line, re.I):
            tone = "ok"
        elif "INFO" in line:
            tone = "info"
        else:
            tone = "text_sec"
        self._log(line, tone)

    def _log(self, text: str, tone: str = "text_sec"):
        col = PAL.get(tone, PAL["text_sec"])
        self.console.append(f'<span style="color:{col.name()};">{text}</span>')
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def _poll_frame(self):
        if not self._frame_pipe or not Path(self._frame_pipe).exists():
            return
        try:
            mtime = Path(self._frame_pipe).stat().st_mtime
        except OSError:
            return
        if mtime <= self._last_frame_mtime:
            return
        self._last_frame_mtime = mtime
        pix = QPixmap(self._frame_pipe)
        if not pix.isNull():
            self._preview.set_pixmap(pix)

    def _toggle_pause(self):
        if not self._process or self._process.state() == QProcess.ProcessState.NotRunning:
            return
        pid = int(self._process.processId())
        if not pid:
            return
        try:
            if self._paused:
                os.kill(pid, signal.SIGCONT)
                self.pause_btn.setText("❚❚ Pause")
                self._log("[INFO] Resumed.", "info")
            else:
                os.kill(pid, signal.SIGSTOP)
                self.pause_btn.setText("▶ Resume")
                self._log("[INFO] Paused.", "warn")
            self._paused = not self._paused
        except (ProcessLookupError, PermissionError, OSError) as exc:
            self._log(f"[WARN] Pause unsupported: {exc}", "warn")
            self.pause_btn.setVisible(False)

    def _stop(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log("[WARN] Stopping…", "warn")
            if self._paused:
                try:
                    os.kill(int(self._process.processId()), signal.SIGCONT)
                except OSError:
                    pass
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()

    def _on_finished(self, exit_code: int, _status):
        if self._frame_timer:
            self._frame_timer.stop(); self._frame_timer = None
        for r in self._rows:
            if r.state == "running":
                r.set_state("done"); r.set_ratio(1.0)
                r.set_sub(f"done · {r.tracks} tracks")
        self._done_count = self._total
        self._cur_ratio = 0.0
        self._update_whole()
        self.job_bar.setValue(1000); self.card_job.set_value("100%")
        self.stop_btn.setEnabled(False)
        if exit_code == 0:
            self._log("[OK] Processing complete.", "ok")
        else:
            self._log(f"[ERROR] Exited with code {exit_code}.", "err")
        self._cleanup_tmp()

    def _cleanup_tmp(self):
        for tmp in (self._frame_pipe,):
            if tmp and Path(tmp).exists():
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


class PreviewWidget(GlassPanel):
    """16:9 live preview with HUD overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled = False
        self._rec = False
        self._fps = 0.0
        self._filename = ""
        self._frame = 0
        self._total = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        self._video = QLabel()
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setMinimumHeight(180)
        self._video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        register_style(self._video, lambda w: w.setStyleSheet(
            f"background:#05050C; border-radius:8px; color:{PAL['text_dim'].name()};"))
        self._video.setText("preview disabled (output.show)")
        v.addWidget(self._video, 1)
        self.setMinimumHeight(220)

    def set_enabled(self, on: bool):
        self._enabled = on
        if not on:
            self._video.setText("preview disabled (output.show)")
            self._video.setPixmap(QPixmap())
        else:
            self._video.setText("waiting for first frame…")
        self.update()

    def set_pixmap(self, pix: QPixmap):
        if not self._enabled:
            return
        scaled = pix.scaled(self._video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self._video.setPixmap(scaled)

    def set_rec(self, on):
        if on != self._rec:
            self._rec = on
            self.update()

    def set_fps(self, fps): self._fps = fps; self.update()
    def set_filename(self, name): self._filename = name; self.update()

    def set_frame_info(self, frame, total):
        self._frame, self._total = frame, total
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if not self._enabled:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = 18
        # REC dot + label
        if self._rec:
            p.setBrush(QBrush(PAL["err"])); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(m, m, 9, 9))
        p.setPen(QPen(PAL["text_pri"]))
        p.setFont(_mono(9, bold=True))
        p.drawText(QRectF(m + 14, m - 4, 80, 16), Qt.AlignmentFlag.AlignVCenter, "REC")
        # FPS top-right
        p.drawText(QRectF(self.width() - 120, m - 4, 100, 16),
                   Qt.AlignmentFlag.AlignRight, f"{self._fps:.1f} FPS")
        # filename bottom-left, frame counter bottom-right
        p.setPen(QPen(PAL["text_sec"]))
        if self._filename:
            p.drawText(QRectF(m, self.height() - m - 14, self.width() - 2 * m, 16),
                       Qt.AlignmentFlag.AlignLeft, Path(self._filename).name)
        if self._total:
            p.drawText(QRectF(m, self.height() - m - 14, self.width() - 2 * m, 16),
                       Qt.AlignmentFlag.AlignRight, f"f {self._frame} / {self._total}")


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PolytrackApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polytrack")
        self.resize(1240, 820)
        self.setMinimumSize(1040, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = HeaderBar(self._goto_stage, self._set_theme)
        root.addWidget(self.header)

        self.stack = QStackedWidget()
        self.setup = SetupScreen(self._on_continue)
        self.processing = ProcessingScreen()
        self.stack.addWidget(self.setup)
        self.stack.addWidget(self.processing)
        root.addWidget(self.stack, 1)

        self._goto_stage(0)

    def _goto_stage(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.header.set_stage(idx)

    def _on_continue(self, config_path: str, config: dict, skip_existing: bool = False):
        self._goto_stage(1)
        self.header.run_pill.set_tone("warn", "● running")
        self.processing.start_run(config_path, config, skip_existing)

    def _set_theme(self, mode: str):
        if THEME:
            THEME.set_mode(mode)


def main():
    global THEME
    app = QApplication(sys.argv)
    app.setApplicationName("Polytrack")
    app.setOrganizationName("Polytrack")

    THEME = ThemeManager(app)
    THEME.apply()

    window = PolytrackApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
