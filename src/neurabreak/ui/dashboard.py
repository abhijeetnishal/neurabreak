"""Analytics dashboard window.

Opens as a separate non-modal window from the tray menu. The window
auto-refreshes every 60 seconds while it's visible so charts stay
current without the user having to do anything.

Charts:
  - Today's posture timeline (good/bad % per hour of the day)
  - Weekly posture score trend (line chart, last 7 days)
  - Break compliance this week (bar chart: triggered vs taken)
  - Session summary stats (today's active time, avg score, breaks etc.)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import  QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from neurabreak.data.journal import HealthJournalService

from neurabreak.ui.branding import apply_window_icon, logo_pixmap

log = structlog.get_logger()

# Colour palette — matches the tray icon colours
_GOOD_COLOUR = "#2ecc71"
_BAD_COLOUR = "#e74c3c"
_NEUTRAL_COLOUR = "#3498db"
_WARNING_COLOUR = "#f39c12"
_MUTED = "#95a5a6"

_REFRESH_INTERVAL_MS = 30_000  # 30-second auto-refresh (keeps "Active Today" current)


def _fmt_active_time(total_secs: int) -> str:
    """Format a duration in seconds for the stat card.

    < 60 s   → "45s"
    < 1 h    → "23m"
    >= 1 h   → "1h 23m"
    """
    if total_secs < 60:
        return f"{total_secs}s"
    if total_secs < 3600:
        return f"{total_secs // 60}m"
    hours = total_secs // 3600
    mins = (total_secs % 3600) // 60
    return f"{hours}h {mins}m"


def _make_stat_card(label: str, value: str, colour: str = _NEUTRAL_COLOUR) -> QFrame:
    """A small rounded card showing one key metric."""
    card = QFrame()
    card.setFrameShape(QFrame.Shape.StyledPanel)
    card.setStyleSheet(f"""
        QFrame {{
            background-color: #2b2b2b;
            border-radius: 8px;
            border: 1px solid #3a3a3a;
        }}
    """)

    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(4)

    value_lbl = QLabel(value)
    value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = QFont()
    font.setPointSize(28)
    font.setBold(True)
    value_lbl.setFont(font)
    value_lbl.setStyleSheet(f"color: {colour}; border: none;")

    label_lbl = QLabel(label)
    label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px; border: none;")

    layout.addWidget(value_lbl)
    layout.addWidget(label_lbl)
    return card


class _HourlyChart(QWidget):
    """Simple bar chart drawn with QPainter — no external chart lib needed.

    Each bar represents one hour 0-23.  Bar height is the percentage of
    good-posture detections in that hour.  Colours: green = good, red = bad.
    Empty hours are shown as a thin grey line so the time axis is always
    fully rendered.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: list[tuple[int, float, float]] = []  # (hour, good_pct, bad_pct)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, hourly_posture: list) -> None:  # list[HourlyPosture]
        self._data = [(h.hour, h.good_pct, h.bad_pct) for h in hourly_posture]
        self.update()

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QBrush, QColor, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        padding = 8
        chart_h = h - 30  # leave room for hour labels at bottom
        bar_w = max(2, (w - 2 * padding) / 24 - 2)

        good_brush = QBrush(QColor(_GOOD_COLOUR))
        bad_brush = QBrush(QColor(_BAD_COLOUR))
        empty_pen = QPen(QColor(_MUTED))
        painter.setPen(Qt.PenStyle.NoPen)

        hour_data = {h_: (g, b) for h_, g, b in self._data}

        for hour in range(24):
            x = padding + hour * ((w - 2 * padding) / 24)
            good_pct, bad_pct = hour_data.get(hour, (0.0, 0.0))

            if good_pct == 0 and bad_pct == 0:
                # Empty slot — draw a thin grey notch
                painter.setPen(empty_pen)
                painter.drawLine(int(x + bar_w / 2), chart_h - 2, int(x + bar_w / 2), chart_h)
                painter.setPen(Qt.PenStyle.NoPen)
                continue

            # Stack good on top of bad
            bad_h = int(chart_h * bad_pct / 100)
            good_h = int(chart_h * good_pct / 100)

            # bad posture at bottom
            painter.setBrush(bad_brush)
            painter.drawRect(int(x), chart_h - bad_h, int(bar_w), bad_h)

            # good posture above bad
            painter.setBrush(good_brush)
            painter.drawRect(int(x), chart_h - bad_h - good_h, int(bar_w), good_h)

        # Hour labels — every 3 hours
        label_pen = QPen(QColor(_MUTED))
        painter.setPen(label_pen)
        small_font = painter.font()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        for hour in range(0, 24, 3):
            x = padding + hour * ((w - 2 * padding) / 24)
            label = f"{hour:02d}"
            painter.drawText(int(x), h - 4, label)

        painter.end()


class _WeeklyChart(QWidget):
    """Line chart for 7-day posture score trend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: list[tuple[date, float]] = []  # (day, score 0–1)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, daily_scores: list) -> None:  # list[DailyScore]
        self._data = [(ds.day, ds.avg_score) for ds in daily_scores]
        self.update()

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
        from PySide6.QtGui import QPainter

        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        pad_x, pad_y = 32, 12
        chart_w = w - 2 * pad_x
        chart_h = h - pad_y * 2 - 18  # room for day labels

        n = len(self._data)
        if n == 0:
            painter.end()
            return

        def _to_xy(i: int, score: float) -> tuple[int, int]:
            x = pad_x + int(i / max(n - 1, 1) * chart_w)
            y = pad_y + int((1 - score) * chart_h)
            return x, y

        # Baseline
        base_pen = QPen(QColor("#3a3a3a"))
        base_pen.setWidth(1)
        painter.setPen(base_pen)
        painter.drawLine(pad_x, pad_y + chart_h, pad_x + chart_w, pad_y + chart_h)

        # Fill area under the line
        path = QPainterPath()
        x0, y0 = _to_xy(0, self._data[0][1])
        path.moveTo(x0, pad_y + chart_h)
        path.lineTo(x0, y0)
        for i, (_, score) in enumerate(self._data[1:], start=1):
            xi, yi = _to_xy(i, score)
            path.lineTo(xi, yi)
        xn, yn = _to_xy(n - 1, self._data[-1][1])
        path.lineTo(xn, pad_y + chart_h)
        path.closeSubpath()
        fill = QColor(_NEUTRAL_COLOUR)
        fill.setAlpha(40)
        painter.fillPath(path, QBrush(fill))

        # Line + dots
        line_pen = QPen(QColor(_NEUTRAL_COLOUR))
        line_pen.setWidth(2)
        painter.setPen(line_pen)
        points = [_to_xy(i, s) for i, (_, s) in enumerate(self._data)]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            painter.drawLine(x1, y1, x2, y2)

        dot_pen = QPen(QColor(_NEUTRAL_COLOUR))
        dot_pen.setWidth(2)
        painter.setPen(dot_pen)
        painter.setBrush(QBrush(QColor("#1e1e1e")))
        for x, y in points:
            painter.drawEllipse(x - 4, y - 4, 8, 8)

        # Day labels
        label_pen = QPen(QColor(_MUTED))
        painter.setPen(label_pen)
        small_font = painter.font()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        for i, (day, _) in enumerate(self._data):
            x, _ = _to_xy(i, 0)
            painter.drawText(x - 12, h - 2, day.strftime("%a %d"))

        painter.end()


class _ComplianceChart(QWidget):
    """Horizontal bar comparing breaks triggered vs taken."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._triggered = 0
        self._taken = 0
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, triggered: int, taken: int) -> None:
        self._triggered = triggered
        self._taken = taken
        self.update()

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QBrush, QColor, QPen
        from PySide6.QtGui import QPainter

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 8
        bar_h = 20
        label_w = 80

        bars = [
            ("Triggered", self._triggered, _WARNING_COLOUR),
            ("Taken",     self._taken,     _GOOD_COLOUR),
        ]
        max_val = max(self._triggered, 1)

        for row, (name, val, colour) in enumerate(bars):
            y = pad + row * (bar_h + 8)
            bar_width = int((w - label_w - 2 * pad) * val / max_val)

            # Background track
            painter.setBrush(QBrush(QColor("#3a3a3a")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(label_w, y, w - label_w - pad, bar_h, 4, 4)

            # Value bar
            painter.setBrush(QBrush(QColor(colour)))
            if bar_width > 0:
                painter.drawRoundedRect(label_w, y, bar_width, bar_h, 4, 4)

            # Label
            painter.setPen(QPen(QColor(_MUTED)))
            small_font = painter.font()
            small_font.setPointSize(9)
            painter.setFont(small_font)
            painter.drawText(pad, y + bar_h - 4, f"{name}")

            # Count
            painter.drawText(label_w + bar_width + 4, y + bar_h - 4, str(val))

        painter.end()


class DashboardWindow(QMainWindow):
    """Main analytics window.

    Charts:
      - Today's posture timeline (good/bad % per hour)
      - Weekly posture score trend
      - Break compliance (breaks taken vs breaks triggered)
      - Session summary stat cards
    """

    def __init__(self, journal: HealthJournalService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._journal = journal

        self.setWindowTitle("NeuraBreak — Health Dashboard")
        apply_window_icon(self)
        self.setMinimumSize(620, 560)
        # Prevent dashboard close from quitting the application
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 6px;
                font-size: 12px;
                color: #aaaaaa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 6px 14px;
                color: #e0e0e0;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:pressed { background-color: #222222; }
            QLabel { color: #e0e0e0; }
            QScrollArea { border: none; }
        """)

        self._build_ui()

        # Auto-refresh while the window is visible
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.setInterval(_REFRESH_INTERVAL_MS)
        self._refresh_timer.start()

    #  UI construction

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header
        header = QHBoxLayout()

        title = QLabel("Health Dashboard")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch()

        self._last_refresh_lbl = QLabel("Refreshing…")
        self._last_refresh_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        header.addWidget(self._last_refresh_lbl)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)

        root.addLayout(header)

        # Stat cards
        self._cards_layout = QGridLayout()
        self._cards_layout.setSpacing(10)

        self._card_active = _make_stat_card("Active Today", "—", _NEUTRAL_COLOUR)
        self._card_score = _make_stat_card("Posture Score", "—", _GOOD_COLOUR)
        self._card_breaks = _make_stat_card("Breaks Taken", "—", _WARNING_COLOUR)
        self._card_streak = _make_stat_card("Good Days Streak", "—", _GOOD_COLOUR)

        self._cards_layout.addWidget(self._card_active, 0, 0)
        self._cards_layout.addWidget(self._card_score, 0, 1)
        self._cards_layout.addWidget(self._card_breaks, 0, 2)
        self._cards_layout.addWidget(self._card_streak, 0, 3)
        root.addLayout(self._cards_layout)

        # Scrollable chart area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        charts_layout = QVBoxLayout(scroll_content)
        charts_layout.setSpacing(12)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

        # Today hours chart
        today_group = QGroupBox("Today — Posture by Hour")
        today_g_layout = QVBoxLayout(today_group)
        self._hourly_chart = _HourlyChart()
        self._hourly_legend = QLabel(
            f'  <span style="color:{_GOOD_COLOUR};">&#9632;</span>'
            f'<span style="color:#e0e0e0;"> Good posture</span>'
            f'&nbsp;&nbsp;&nbsp;'
            f'<span style="color:{_BAD_COLOUR};">&#9632;</span>'
            f'<span style="color:#e0e0e0;"> Bad posture</span>'
        )
        self._hourly_legend.setTextFormat(Qt.TextFormat.RichText)
        self._hourly_legend.setStyleSheet("font-size: 10px;")
        today_g_layout.addWidget(self._hourly_chart)
        today_g_layout.addWidget(self._hourly_legend)
        charts_layout.addWidget(today_group)

        # Weekly trend chart
        week_group = QGroupBox("This Week — Posture Score Trend")
        week_g_layout = QVBoxLayout(week_group)
        self._weekly_chart = _WeeklyChart()
        week_g_layout.addWidget(self._weekly_chart)
        charts_layout.addWidget(week_group)

        # Break compliance chart
        compliance_group = QGroupBox("Break Compliance — Last 7 Days")
        comp_g_layout = QVBoxLayout(compliance_group)
        self._compliance_chart = _ComplianceChart()
        comp_g_layout.addWidget(self._compliance_chart)
        charts_layout.addWidget(compliance_group)

        charts_layout.addStretch()

        # Export buttons
        export_row = QHBoxLayout()
        export_row.addStretch()

        export_csv_btn = QPushButton("Export CSV…")
        export_csv_btn.clicked.connect(self._on_export_csv)
        export_row.addWidget(export_csv_btn)

        export_json_btn = QPushButton("Export JSON…")
        export_json_btn.clicked.connect(self._on_export_json)
        export_row.addWidget(export_json_btn)

        root.addLayout(export_row)

    # Public API

    def show(self) -> None:  # type: ignore[override]
        super().show()
        self.activateWindow()
        self.raise_()
        self.refresh()

    def refresh(self) -> None:
        """Re-query the journal and redraw all charts."""
        try:
            self._refresh_stat_cards()
            self._refresh_hourly_chart()
            self._refresh_weekly_chart()
            self._refresh_compliance_chart()

            from PySide6.QtCore import QDateTime
            ts = QDateTime.currentDateTime().toString("hh:mm:ss")
            self._last_refresh_lbl.setText(f"Updated {ts}")
        except Exception as exc:
            log.error("dashboard_refresh_failed", error=str(exc), exc_info=True)

    # Internal refresh helpers

    def _refresh_stat_cards(self) -> None:
        summary = self._journal.get_today_summary()

        if summary:
            active_text = _fmt_active_time(summary.total_active_secs)
            score_text = f"{int(summary.avg_posture_score * 100)}%"
            breaks_text = str(summary.breaks_taken)
        else:
            active_text = "0s"
            score_text = "—"
            breaks_text = "0"

        # Update value labels in each card
        def _set_card_value(card: QFrame, text: str) -> None:
            for lbl in card.findChildren(QLabel):
                if lbl.font().bold() and lbl.font().pointSize() >= 20:
                    lbl.setText(text)
                    break

        _set_card_value(self._card_active, active_text)
        _set_card_value(self._card_score, score_text)
        _set_card_value(self._card_breaks, breaks_text)

        # Good days streak — count consecutive days with score > 70%
        daily = self._journal.get_daily_scores(days=30)
        streak = 0
        for ds in reversed(daily):
            if ds.avg_score >= 0.7:
                streak += 1
            else:
                break
        _set_card_value(self._card_streak, str(streak))

    def _refresh_hourly_chart(self) -> None:
        hourly = self._journal.get_hourly_posture_today()
        self._hourly_chart.set_data(hourly)

    def _refresh_weekly_chart(self) -> None:
        daily = self._journal.get_daily_scores(days=7)
        self._weekly_chart.set_data(daily)

    def _refresh_compliance_chart(self) -> None:
        stats = self._journal.get_break_compliance(days=7)
        self._compliance_chart.set_data(
            triggered=stats["triggered"],
            taken=stats["taken"],
        )

    #  Export handlers

    def _on_export_csv(self) -> None:
        default_name = f"neurabreak_{date.today().isoformat()}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export data as CSV",
            str(Path.home() / "Downloads" / default_name),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self._journal.export_csv(path)
            QMessageBox.information(self, "Export complete", f"Data saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_export_json(self) -> None:
        default_name = f"neurabreak_{date.today().isoformat()}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export data as JSON",
            str(Path.home() / "Downloads" / default_name),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            self._journal.export_json(path)
            QMessageBox.information(self, "Export complete", f"Data saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def closeEvent(self, event) -> None:
        # Stop the auto-refresh timer when hidden — no wasted cycles
        self._refresh_timer.stop()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        self._refresh_timer.start()
        super().showEvent(event)

