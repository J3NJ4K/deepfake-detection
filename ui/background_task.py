"""
Run a slow function (e.g. the deepfake analysis) in a separate thread, so the window
stays responsive, then get the result back in the interface.

    self.task = BackgroundTask(self)
    self.task.start(analyze_deepfake, (image,), self._on_done, self._on_error)

IMPORTANT: on_done and on_error must be METHODS of a Qt object living in the interface
(e.g. methods of MainWindow), not lambdas: Qt then calls them in the interface thread,
which is the only place where widgets may be touched.
"""

from PySide6.QtCore import QObject, QThread, Signal


class _Worker(QObject):
    finished = Signal(object)   # result of the function
    failed = Signal(str)        # error message

    def __init__(self, func, args):
        super().__init__()
        self._func = func
        self._args = args

    def run(self):
        try:
            result = self._func(*self._args)
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class BackgroundTask(QObject):
    """Runs one function at a time in its own thread."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None

    def is_running(self) -> bool:
        return self._thread is not None

    def start(self, func, args, on_done, on_error) -> None:
        if self.is_running():
            raise RuntimeError("A background task is already running")
        thread = QThread()
        worker = _Worker(func, args)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(on_done)
        worker.failed.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)

        # Keep references, otherwise Python would delete them while the thread runs
        self._thread, self._worker = thread, worker
        thread.start()

    def wait(self) -> None:
        """Block until the running task is finished (used when the window closes)."""
        if self._thread is not None:
            self._thread.wait()

    def _on_thread_finished(self) -> None:
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = self._worker = None
