# Pc/vision/camera_io.py
"""
Módulo para abrir la cámara según los parámetros del archivo YAML (config.yaml).
Usa OpenCV y devuelve un manejador sencillo con .read(), .snapshot() y .release().
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import cv2
import numpy as np

from Pc.common.config_loader import load_config

# Backends de OpenCV
_BACKENDS = {
    "v4l2": getattr(cv2, "CAP_V4L2", 200),  # Linux
    "any": 0,                              # OpenCV elige el mejor
}


def _fourcc(code: Optional[str]) -> Optional[int]:
    """Convierte string 'MJPG' o 'YUYV' a código FOURCC."""
    if not code:
        return None
    code = code.strip().upper()
    if len(code) != 4:
        return None
    return cv2.VideoWriter_fourcc(*code)


def _idx_from_device_path(dev: Optional[str]) -> Optional[int]:
    """Extrae el índice desde '/dev/videoX'."""
    if not dev:
        return None
    m = re.search(r"/dev/video(\d+)$", dev.strip())
    return int(m.group(1)) if m else None


@dataclass
class CameraHandle:
    """Objeto de alto nivel para operar la cámara."""
    cap: cv2.VideoCapture
    flip_h: bool
    flip_v: bool
    rotate_deg: int

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, np.ndarray | None]:
        """Lee un frame aplicando flip/rotación según config."""
        t0 = time.time()
        while True:
            ok, frame = self.cap.read()
            if ok:
                frame = self._postprocess(frame)
                return True, frame
            if (time.time() - t0) * 1000.0 > timeout_ms:
                return False, None
            time.sleep(0.01)

    def snapshot(self, out_path: Path) -> bool:
        """Guarda una imagen actual en el path indicado."""
        ok, frame = self.read()
        if ok:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            return cv2.imwrite(str(out_path), frame)
        return False

    def release(self) -> None:
        """Libera la cámara."""
        try:
            self.cap.release()
        except Exception:
            pass

    def _postprocess(self, frame: np.ndarray) -> np.ndarray:
        """Aplica flips y rotación si se configuraron."""
        if self.flip_h:
            frame = cv2.flip(frame, 1)
        if self.flip_v:
            frame = cv2.flip(frame, 0)
        if self.rotate_deg in (90, 180, 270):
            if self.rotate_deg == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate_deg == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


def open_camera_from_config() -> CameraHandle:
    """Abre la cámara leyendo parámetros desde Data/config.yaml."""
    cfg = load_config()
    cam = cfg.get("camera", {}) or {}

    backend = _BACKENDS.get(str(cam.get("backend", "any")).lower(), 0)
    idx = _idx_from_device_path(cam.get("device"))
    if idx is None:
        idx = int(cam.get("index", 0))

    width  = int(cam.get("width", 640))
    height = int(cam.get("height", 480))
    fps    = int(cam.get("fps", 30))
    buffer_size = int(cam.get("buffer_size", 0))
    warmup_ms = int(cam.get("warmup_ms", 300))
    open_retries = max(1, int(cam.get("open_retries", 1)))
    read_timeout_ms = int(cam.get("read_timeout_ms", 1000))
    fourcc = _fourcc(cam.get("fourcc"))

    flip_h = bool(cam.get("flip_h", False))
    flip_v = bool(cam.get("flip_v", False))
    rotate_deg = int(cam.get("rotate_deg", 0)) % 360

    last_err = None
    for attempt in range(1, open_retries + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            last_err = f"no se pudo abrir idx={idx} (backend={backend})"
            time.sleep(0.2)
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if buffer_size > 0 and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if fourcc is not None and hasattr(cv2, "CAP_PROP_FOURCC"):
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        time.sleep(warmup_ms / 1000.0)
        ok, _ = cap.read()
        if ok:
            handle = CameraHandle(cap=cap, flip_h=flip_h, flip_v=flip_v, rotate_deg=rotate_deg)
            ok2, _ = handle.read(timeout_ms=read_timeout_ms)
            if ok2:
                return handle
            else:
                last_err = "timeout leyendo frame tras abrir"
        else:
            last_err = "no devolvió frame tras abrir"

        cap.release()
        time.sleep(0.2)

    raise RuntimeError(f"No fue posible abrir la cámara: {last_err}")
