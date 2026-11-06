"""
Side panel with the results of the last detection, instead of pop-up windows:

    Detection results                 ×
    Likely AI-generated · 2 faces, 1 likely manipulated
    ┌──────────────────────────────┐
    │ Whole image    AI-generated  │   <- UnivFD (whole picture)
    │ ██████████████░░  87% AI     │
    └──────────────────────────────┘
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

    def __init__(self, title: str, verdict: str = "", color: str = "", probability=None,
                 bar_format: str = "%p% fake"):
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
            bar.setFormat(bar_format)
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

        # Whole-image result (AI-generated or not), only for the deepfake analysis
        self.image_area = QWidget()
        self.image_area_layout = QVBoxLayout(self.image_area)
        self.image_area_layout.setContentsMargins(0, 0, 0, 0)
        self.image_area.setObjectName("ImageArea")
        self.image_area.setStyleSheet("#ImageArea { border: 1px solid palette(mid); border-radius: 4px; }")
        self.image_area.hide()
        self._image_row = None
        layout.addWidget(self.image_area)

        self.faces_title = QLabel("Faces")
        self.faces_title.setEnabled(False)
        self.faces_title.hide()
        layout.addWidget(self.faces_title)

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
        self._set_image_row(None)
        self.faces_title.hide()
        self.summary.setText("Run a detection to see the results here.")
        self.note.setText("")

    def _set_image_row(self, row) -> None:
        """Replace the whole-image result (None = hide it)."""
        if self._image_row is not None:
            self.image_area_layout.removeWidget(self._image_row)
            self._image_row.deleteLater()
        self._image_row = row
        if row is None:
            self.image_area.hide()
        else:
            self.image_area_layout.addWidget(row)
            self.image_area.show()

    def show_faces(self, boxes) -> None:
        """Results of the face detection: list of (x1, y1, x2, y2)."""
        self._set_image_row(None)
        self.faces_title.hide()
        self.list.blockSignals(True)
        self.list.clear()
        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            self._add_row(_FaceRow(f"Face {i}", f"{x2 - x1}×{y2 - y1} px", FACE_COLOR))
        self.list.blockSignals(False)
        n = len(boxes)
        self.summary.setText("No faces detected." if n == 0 else f"{n} face{'s' if n > 1 else ''} detected")
        self.note.setText("Click a face to show it on the image." if n else "")

    def show_deepfake(self, faces, generated_probability=None, threshold: float = 0.5) -> None:
        """
        Results of the deepfake analysis:
        - generated_probability: probability that the WHOLE image is AI-generated (UnivFD)
        - faces: list of {"bbox", "manipulation_probability", ...} (Xception, one per face)
        """
        # --- whole image
        parts = []
        if generated_probability is None:
            self._set_image_row(None)
        else:
            is_ai = generated_probability >= threshold
            self._set_image_row(_FaceRow("Whole image", "AI-generated" if is_ai else "Not AI-generated",
                                         FAKE_COLOR if is_ai else REAL_COLOR, generated_probability,
                                         "%p% AI-generated"))
            parts.append("Likely AI-generated" if is_ai else "Likely not AI-generated")

        # --- faces
        self.list.blockSignals(True)
        self.list.clear()
        fakes = 0
        for i, res in enumerate(faces, start=1):
            prob = res["manipulation_probability"]
            is_fake = prob >= 0.5
            fakes += is_fake
            self._add_row(_FaceRow(f"Face {i}", "Fake" if is_fake else "Real",
                                   FAKE_COLOR if is_fake else REAL_COLOR, prob))
        self.list.blockSignals(False)
        n = len(faces)
        self.faces_title.setVisible(n > 0)
        if n == 0:
            parts.append("no faces found")
        else:
            parts.append(f"{n} face{'s' if n > 1 else ''}, {fakes} likely manipulated")
        summary = " · ".join(parts)
        self.summary.setText(summary[0].upper() + summary[1:])

        notes = []
        if generated_probability is not None:
            notes.append(f"Whole image: probability that the picture was created by an AI "
                         f"(above {threshold * 100:.0f}% = likely AI-generated).")
        if n:
            notes.append("Faces: probability that the face was manipulated (above 50% = likely fake). "
                         "Click a face to show it on the image.")
        notes.append("Automatic estimates: they can be wrong.")
        self.note.setText(" ".join(notes))

    # ------------------------------------------------------------------
    def _add_row(self, row: QWidget) -> None:
        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, row)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.face_selected.emit(row)
