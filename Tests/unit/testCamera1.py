#!/usr/bin/env python3
import cv2
import time
import sys

# ======================================================
# 🔧 CONFIGURACIÓN RÁPIDA — SELECCIONA AQUÍ TU CÁMARA
CAMERA_SOURCE = "/dev/video2"      # 👉 Cambia este número (0, 1, 2...) o "/dev/video1"
# ======================================================

def main():
    print("="*60)   
    print(f"🎥 Iniciando cámara desde fuente: {CAMERA_SOURCE}")
    print("="*60)

    # Abrir cámara
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {CAMERA_SOURCE}")
        print("👉 Prueba con otro índice (0, 1, 2...) o verifica con:")
        print("   ls /dev/video*   o   v4l2-ctl --list-devices")
        sys.exit(1)

    # Configuración inicial (puedes modificar resolución o FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Lectura de parámetros reportados
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"📏 Resolución: {width}x{height}  🎞️ FPS reportados: {fps:.1f}")

    print("Controles: [q/ESC] salir | [s] guardar foto | [+/-] cambiar resolución\n")

    prev_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ No se pudo leer frame. Reintentando...")
            time.sleep(0.05)
            continue

        # Calcular FPS en tiempo real
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps_live = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0
            cv2.setWindowTitle("CAM LIVE", f"CAM LIVE ({width}x{height}) ~{fps_live:.1f} FPS")

        # Mostrar imagen
        cv2.imshow("CAM LIVE", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):   # ESC o q
            print("👋 Saliendo...")
            break
        elif key == ord('s'):       # snapshot
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Imagen guardada: {filename}")
        elif key in (ord('+'), ord('=')):   # subir resolución
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"🔺 Resolución aumentada a: {width}x{height}")
        elif key in (ord('-'), ord('_')):   # bajar resolución
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, max(160, width // 2))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max(120, height // 2))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"🔻 Resolución reducida a: {width}x{height}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
