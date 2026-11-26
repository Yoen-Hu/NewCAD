"""ML-side JSON IPC for disposable and reusable CAD workers."""

import json
import os
from pathlib import Path
import subprocess
from queue import Empty, Queue
from threading import Lock, Thread
import time

from .eg_pir_schema import parse_etc_cq, validate_cadquery


REPOSITORY = Path(__file__).resolve().parents[2]
CAD_PYTHON = REPOSITORY.parent / "reproduction" / "cad" / "python.exe"
TIMEOUT_SECONDS = 5.0


def _worker_env(interpreter):
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    prefixes = (interpreter.parent, interpreter.parent / "Library" / "bin", interpreter.parent / "Scripts")
    env["PATH"] = os.pathsep.join([*(str(path) for path in prefixes), env.get("PATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def _invoke(payload, interpreter):
    interpreter = Path(interpreter).resolve()
    env = _worker_env(interpreter)
    started = time.monotonic()
    try:
        run = subprocess.run(
            [str(interpreter), "-m", "src.topocq.sandbox_worker_ocp"],
            input=json.dumps(payload, allow_nan=False), capture_output=True, encoding="utf-8",
            errors="replace", cwd=REPOSITORY, env=env, timeout=TIMEOUT_SECONDS,
        )
        if run.returncode:
            report = {"ok": False, "status": "worker_exit", "returncode": run.returncode,
                      "detail": run.stderr[-4000:]}
        else:
            try:
                report = json.loads(run.stdout)
                if not isinstance(report, dict) or "status" not in report:
                    raise ValueError("worker response lacks status")
            except ValueError as exc:
                report = {"ok": False, "status": "invalid_ipc", "detail": str(exc),
                          "stderr": run.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        report = {"ok": False, "status": "timeout", "timeout_seconds": TIMEOUT_SECONDS}
    except OSError as exc:
        report = {"ok": False, "status": "launch_error", "detail": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def _trial(invoke, prediction, reference=None, *, cad_python=None, align=True, project=True):
    """Validate complete ETC-CQ blocks and evaluate in CAD_PYTHON.

    Reference may be a second ETC-CQ response or an explicit local STEP Path.
    Projection polylines stay visible/hidden separately for downstream rendering.
    """
    try:
        parse_etc_cq(prediction)
        if isinstance(reference, str):
            parse_etc_cq(reference)
        if isinstance(reference, Path):
            reference = reference.resolve(strict=True)
            if reference.suffix.lower() not in {".step", ".stp"}:
                raise ValueError("reference file must be STEP")
    except (ValueError, OSError, OverflowError, RecursionError) as exc:
        return {"ok": False, "status": "static_rejected", "detail": str(exc), "elapsed_seconds": 0.0}
    return invoke({"prediction": prediction, "reference": str(reference) if reference else None,
                    "reference_step": isinstance(reference, Path), "align": align, "project": project},
                   cad_python or os.environ.get("CAD_PYTHON") or CAD_PYTHON)


def _script(invoke, source, *, cad_python=None, project=True):
    """Evaluate a canonical CadQuery-only engineering fixture."""
    try:
        validate_cadquery(source)
    except (ValueError, OverflowError, RecursionError) as exc:
        return {"ok": False, "status": "static_rejected", "detail": str(exc), "elapsed_seconds": 0.0}
    return invoke({"source": source, "project": project},
                   cad_python or os.environ.get("CAD_PYTHON") or CAD_PYTHON)



def run_trial(prediction, reference=None, *, cad_python=None, align=True, project=True):
    """Evaluate ETC-CQ using a disposable CAD subprocess."""
    return _trial(_invoke, prediction, reference, cad_python=cad_python, align=align, project=project)


def run_script(source, *, cad_python=None, project=True):
    """Evaluate a CadQuery fixture using a disposable CAD subprocess."""
    return _script(_invoke, source, cad_python=cad_python, project=project)


class _WorkerSlot:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.process = None
        self.completed = 0

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
        self.completed = 0

    def invoke(self, payload):
        started = time.monotonic()
        if self.process is None:
            self.process = subprocess.Popen(
                [str(self.interpreter), "-u", "-m", "src.topocq.sandbox_worker_ocp", "--persistent"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=REPOSITORY, env=_worker_env(self.interpreter),
            )
        process = self.process
        response = Queue(maxsize=1)
        encoded = (json.dumps(payload, allow_nan=False) + "\n").encode("ascii")

        def exchange():
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
                response.put(process.stdout.readline())
            except (OSError, ValueError) as exc:
                response.put(exc)

        io = Thread(target=exchange, daemon=True)
        io.start()
        try:
            value = response.get(timeout=max(0, TIMEOUT_SECONDS - (time.monotonic() - started)))
        except Empty:
            self.close()
            io.join()
            return {"ok": False, "status": "timeout", "timeout_seconds": TIMEOUT_SECONDS,
                    "elapsed_seconds": time.monotonic() - started, "worker_pid": process.pid}

        io.join()
        if not value or isinstance(value, Exception):
            self.close()
            report = {"ok": False, "status": "worker_exit", "returncode": process.returncode,
                      "detail": str(value) if value else "worker closed its JSON output"}
        else:
            try:
                report = json.loads(value)
                if not isinstance(report, dict) or "status" not in report:
                    raise ValueError("worker response lacks status")
            except ValueError as exc:
                self.close()
                report = {"ok": False, "status": "invalid_ipc", "detail": str(exc)}
            else:
                if payload != {"ping": True}:
                    self.completed += 1
                    if self.completed == 20:
                        self.close()
        report.update(elapsed_seconds=time.monotonic() - started, worker_pid=process.pid)
        return report


class SandboxPool:
    """Reusable CAD subprocess slots, safe for concurrent callers.

    Each admitted task gets five seconds, including cold start if needed.
    A failed or retired slot launches a fresh process on its next task.
    """

    def __init__(self, workers=1, *, cad_python=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.interpreter = Path(cad_python or os.environ.get("CAD_PYTHON") or CAD_PYTHON).resolve()
        self._slots = Queue()
        self._size = workers
        self._state = Lock()
        self._closed = False
        for _ in range(workers):
            self._slots.put(_WorkerSlot(self.interpreter))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._state:
            if self._closed:
                return
            self._closed = True
        for _ in range(self._size):
            self._slots.get().close()

    def _invoke(self, payload, _interpreter):
        with self._state:
            if self._closed:
                raise RuntimeError("sandbox pool is closed")
            slot = self._slots.get()
        try:
            return slot.invoke(payload)
        except OSError as exc:
            slot.close()
            return {"ok": False, "status": "launch_error", "detail": str(exc)}
        finally:
            self._slots.put(slot)

    def warm(self):
        """Initialize one slot and return its measured startup time."""
        return self._invoke({"ping": True}, self.interpreter)

    def run_trial(self, prediction, reference=None, *, align=True, project=True):
        return _trial(self._invoke, prediction, reference, cad_python=self.interpreter, align=align, project=project)

    def run_script(self, source, *, project=True):
        return _script(self._invoke, source, cad_python=self.interpreter, project=project)
