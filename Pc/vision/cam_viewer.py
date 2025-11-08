# Pc/vision/cam_viewer.py
"""
Viewer en tiempo real de la cámara + snapshot con tecla 's'.

Teclas:
  q  -> salir
  s  -> guardar snapshot en paths.snapshots_dir
  g  -> alternar grilla 8x8 (útil para centrar el tablero)
  i  -> mostrar/ocultar info (resolución / FPS)
"""

from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import sys

# --- permitir importar desde la raíz del repo (si lo corres como script) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------------

from Pc.common.config_loader import get_path, load_config
from Pc.vision.cam_io import open_camera_from_config


def _put_text(img, text, org=(10, 24)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def _draw_grid(img, cells=8, color=(0, 255, 0), thickness=1):
    h, w = img.shape[:2]
    dx = w // cells
    dy = h // cells
    for x in range(dx, w, dx):
        cv2.line(img, (x, 0), (x, h), color, thickness)
    for y in range(dy, h, dy):
        cv2.line(img, (0, y), (w, y), color, thickness)

def main():
    cfg = load_config()
    snaps_dir = Path(get_path("snapshots_dir"))
    snaps_dir.mkdir(parents=True, exist_ok=True)

    cam = open_camera_from_config()
    print("✅ Cámara abierta. Teclas: [q]=salir  [s]=snapshot  [i]=info  [g]=grilla")
    show_info = True
    show_grid = False

    t_last = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cam.read(timeout_ms=int(cfg.get("camera", {}).get("read_timeout_ms", 1000)))
            if not ok or frame is None:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                _put_text(blank, "❌ Timeout leyendo frame", (10, 120))
                cv2.imshow("CamViewer", blank)
            else:
                frames += 1
                now = time.time()
                if now - t_last >= 1.0:
                    fps = frames / (now - t_last)
                    t_last = now
                    frames = 0

                view = frame.copy()

                if show_grid:
                    _draw_grid(view, cells=8)
                if show_info:
                    h, w = view.shape[:2]
                    _put_text(view, f"{w}x{h} | {fps:.1f} FPS")

                cv2.imshow("CamViewer", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                out = snaps_dir / f"snapshot_{ts}.jpg"
                if frame is not None:
                    cv2.imwrite(str(out), frame)
                    print(f"💾 Snapshot guardado: {out}")
            elif key == ord('i'):
                show_info = not show_info
            elif key == ord('g'):
                show_grid = not show_grid

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("👋 Cámara liberada. Ventanas cerradas.")


if __name__ == "__main__":
    main()
