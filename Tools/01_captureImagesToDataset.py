# Codigo para formar el DataSet de los intrinsecos
# Tablero vacio orientado en diferentes posiciones

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime

# ========= CONFIGURACIÓN =========
CAMERA_ID       = "/dev/video2"                # <-- tu cámara "1"
FRAME_SIZE      = (1280, 720)      # (ancho, alto) – usa la que vayas a emplear
FPS_TARGET      = 30

CHESSBOARD_SIZE = (7, 7)           # (esquinas internas en columnas, filas)
SQUARE_SIZE_MM  = 40.25             # tamaño de casilla en mm (solo meta-dato aquí)

OUTPUT_ROOT     = "calib_capturas" # carpeta base

# Intentos de fijar enfoque/exposición (pueden no aplicar según driver)
AUTOFOCUS_OFF   = True
AUTOEXPO_OFF    = True
# =================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def draw_text(img, text, org, scale=0.6, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main():
    # Carpeta con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"session_{ts}")
    ensure_dir(out_dir)

    # Guardar metadatos de la sesión
    meta = {
        "camera_id": CAMERA_ID,
        "frame_size": FRAME_SIZE,
        "fps_target": FPS_TARGET,
        "chessboard_size": CHESSBOARD_SIZE,
        "square_size_mm": SQUARE_SIZE_MM,
        "created": ts,
        "notes": "Imágenes para calibración de intrínsecos. Cámara fija."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara", CAMERA_ID)
        return

    # Ajustes de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    # Intentar desactivar autofocus y autoexposición (según backend)
    if AUTOFOCUS_OFF:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if AUTOEXPO_OFF:
        # Algunos drivers usan 0.25 (manual) / 0.75 (auto) en CAP_PROP_AUTO_EXPOSURE
        # Otros usan 1 (manual) / 3 (auto). Probamos ambos.
        if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # Criterios para refinar esquinas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    count = 0
    auto_mode = False
    last_save_time = 0.0
    min_interval_auto = 0.8  # seg entre auto-capturas para evitar duplicados

    print("Controles: 'c'=capturar  'a'=auto-captura ON/OFF  'q'=salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Frame no disponible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar patrón
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        vis = frame.copy()
        status = "Patrón: NO"
        if found:
            # Refinar esquinas
            corners_ref = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1), criteria
            )
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_ref, True)
            status = "Patrón: SÍ"
        draw_text(vis, f"{status} | Capturas: {count}", (10,30))
        draw_text(vis, "c=guardar  a=auto  q=salir", (10,60), scale=0.55, color=(255,255,255))

        cv2.imshow("CALIB - Captura", vis)
        key = cv2.waitKey(1) & 0xFF

        # Modo auto-captura
        now = time.time()
        if auto_mode and found and (now - last_save_time) > min_interval_auto:
            filename = os.path.join(out_dir, f"img_{count:04d}.png")
            cv2.imwrite(filename, frame)
            print(f"📸 [AUTO] guardado: {filename}")
            count += 1
            last_save_time = now

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
            print("🔁 Auto-captura:", "ON" if auto_mode else "OFF")
        elif key == ord('c'):
            if found:
                filename = os.path.join(out_dir, f"img_{count:04d}.png")
                cv2.imwrite(filename, frame)
                print(f"💾 guardado: {filename}")
                count += 1
            else:
                print("❗ No se detecta el patrón completo. Acomoda el tablero y vuelve a intentar.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Sesión finalizada. Imágenes guardadas en: {out_dir}")
    print("Siguiente paso: usar estas imágenes en el script de calibración para obtener K y distorsión.")

if __name__ == "__main__":
    main()
