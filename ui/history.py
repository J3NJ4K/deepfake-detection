import numpy as np


class ImageHistory:
    """
    Undo/redo history for the edited image.

    Stores previous versions of the image (NumPy RGB uint8) in two stacks.
    The functions in core/ always return a NEW array, so storing references
    is safe (no copy needed).
    """

    def __init__(self, max_steps: int = 20):
        # Limit the number of steps so memory doesn't explode with big images
        self.max_steps = max_steps
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    def push(self, image: np.ndarray) -> None:
        """Save the image BEFORE a modification. Clears the redo stack."""
        self._undo_stack.append(image)
        if len(self._undo_stack) > self.max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, current: np.ndarray) -> np.ndarray | None:
        """Return the previous image, and keep the current one for redo."""
        if not self._undo_stack:
            return None
        self._redo_stack.append(current)
        return self._undo_stack.pop()

    def redo(self, current: np.ndarray) -> np.ndarray | None:
        """Return the next image, and keep the current one for undo."""
        if not self._redo_stack:
            return None
        self._undo_stack.append(current)
        return self._redo_stack.pop()

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)
