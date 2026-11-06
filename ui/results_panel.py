"""
Side panel with the results of the last detection, instead of pop-up windows:

    Detection results                 ×
    2 faces · 1 likely fake
    ┌──────────────────────────────┐
    │ Face 1                 Fake  │
    │ ██████████████░░░░  78% fake │
    ├──────────────────────────────┤
    │ Face 2                 Real  │
    │ ███░░░░░░░░░░░░░░░  12% fake │
    └──────────────────────────────┘
    Probability that the face was manipulated. Above 50% = likely fake.
    Automatic estimate: it can be wrong.

Clicking a face emits face_selected(index) so MainWindow can show it on the image.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
    QToolButton, QVBoxLayout, QWidget,
)

FAKE_COLOR = "#d93b3b"
REAL_COLOR = "#2e9e4f"
FACE_COLOR = "#0096ff"   # same blue as the boxes of the face detection


class _FaceRow(QWidget):
    """One face of the list: name + verdict, and a probability bar (if any)."""

    def __init__(self, title: str, verdict: str = "", color: str = "", probability=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.addWidget(QLabel(title))
        header.addStretch()
        if verdict:
            label = QLabel(verdict)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            label.setStyleSheet(f"color: {color};")
            header.addWidget(label)
        layout.addLayout(header)

        if probability is not None:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round(probability * 100))
            bar.setFormat("%p% fake")
            bar.setFixedHeight(14)
            bar.setStyleSheet(f"QProgressBar {{ border: 1px solid palette(mid); border-radius: 3px;"
                              f" text-align: center; font-size: 10px; }}"
                              f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}")
            layout.addWidget(bar)


class ResultsPanel(QWidget):
    face_selected = Signal(int)   # index of the clicked face
    closed = Signal()             # the × button was pressed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ResultsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#ResultsPanel { border-left: 1px solid palette(mid); }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)

        # --- title + close button
        top = QHBoxLayout()
        title = QLabel("Detection results")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        top.addWidget(title)
        top.addStretch()
        close = QToolButton()
        close.setText("×")
        close.setAutoRaise(True)
        close.setToolTip("Hide the results")
        close.clicked.connect(self.closed.emit)
        top.addWidget(close)
        layout.addLayout(top)

        self.summary = QLabel("Run a detection to see the results here.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list, 1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setEnabled(False)   # grey text
        layout.addWidget(self.note)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.list.clear()
        self.summary.setText("Run a detection to see the results here.")
        self.note.setText("")

    def show_faces(self, boxes) -> None:
        """Results of the face detection: list of (x1, y1, x2, y2)."""
        self.list.blockSignals(True)
        self.list.clear()
        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            self._add_row(_FaceRow(f"Face {i}", f"{x2 - x1}×{y2 - y1} px", FACE_COLOR))
        self.list.blockSignals(False)
        n = len(boxes)
        self.summary.setText("No faces detected." if n == 0 else f"{n} face{'s' if n > 1 else ''} detected")
        self.note.setText("Click a face to show it on the image." if n else "")

    def show_deepfake(self, results) -> None:
        """Results of the deepfake analysis: list of {"bbox", "manipulation_probability"}."""
        self.list.blockSignals(True)
        self.list.clear()
        fakes = 0
        for i, res in enumerate(results, start=1):
            prob = res["manipulation_probability"]
            is_fake = prob >= 0.5
            fakes += is_fake
            self._add_row(_FaceRow(f"Face {i}", "Fake" if is_fake else "Real",
                                   FAKE_COLOR if is_fake else REAL_COLOR, prob))
        self.list.blockSignals(False)
        n = len(results)
        if n == 0:
            self.summary.setText("No faces detected to analyse.")
            self.note.setText("")
        else:
            self.summary.setText(f"{n} face{'s' if n > 1 else ''} · "
                                 f"{fakes} likely fake")
            self.note.setText("Bar = probability that the face was manipulated "
                              "(above 50% = likely fake). Automatic estimate: it can be wrong. "
                              "Click a face to show it on the image.")

    # ------------------------------------------------------------------
    def _add_row(self, row: QWidget) -> None:
        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, row)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.face_selected.emit(row)
