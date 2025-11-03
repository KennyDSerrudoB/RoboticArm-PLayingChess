# tests/unit/test_opencv_full.py
# Ejecutable standalone (sin pytest). Corre todo o partes con flags.
# Ejemplos:
#   python tests/unit/test_opencv_full.py --all
#   python tests/unit/test_opencv_full.py --camera
#   CAM_INDEX=2 python tests/unit/test_opencv_full.py --camera

import os
import sys
import time
import math
import argparse
import tempfile
import numpy as np
import cv2

def _log(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# ---------------------------
# Secciones de prueba
# ---------------------------

def test_00_version_buildinfo():
    _log("OpenCV — versión y build info")
    ver = cv2.__version__
    print("cv2.__version__:", ver)
    if not (isinstance(ver, str) and len(ver) >= 3):
        raise AssertionError("Versión de OpenCV no válida")

    try:
        build = cv2.getBuildInformation()
        print("Build info (primeros 300 chars):", build[:300].replace("\n", " "))
        if not ("General configuration" in build or len(build) > 100):
            raise AssertionError("Build info inesperada")
    except Exception as e:
        print("getBuildInformation() no disponible:", e)

def test_01_video_backends_and_cuda():
    _log("Backends de video y soporte CUDA")
    try:
        backs = cv2.videoio_registry.getBackends()
        names = [cv2.videoio_registry.getBackendName(b) for b in backs]
        print("Backends disponibles:", names)
    except Exception as e:
        print("videoio_registry no disponible:", e)

    if hasattr(cv2, "cuda"):
        try:
            ndev = cv2.cuda.getCudaEnabledDeviceCount()
            print("CUDA devices:", ndev)
        except Exception as e:
            print("Consulta CUDA falló:", e)
    else:
        print("OpenCV sin módulo CUDA")

def test_02_camera_capture_if_available():
    _log("Cámara (si existe)")
    # 1) Preferir índice por variable de entorno
    env_idx = os.getenv("CAM_INDEX")
    preferred = []
    if env_idx is not None and env_idx.strip() != "":
        try:
            preferred.append(int(env_idx))
        except ValueError:
            preferred.append(env_idx)  # permitir /dev/videoX como string

    # 2) Agregar un barrido básico 0..3 como fallback
    preferred.extend([0, 1, 2, 3])

    cap = None
    used_idx = None

    for idx in preferred:
        if isinstance(idx, str) and idx.startswith("/dev/video"):
            c = cv2.VideoCapture(idx)
        else:
            c = cv2.VideoCapture(int(idx))
        ok, frame = c.read()
        if ok and frame is not None:
            cap = c
            used_idx = idx
            break
        c.release()

    if cap is None:
        print("No se encontró cámara. Se omite este bloque.")
        return

    # Configurar props (no siempre respetadas por el driver)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, frame = cap.read()
    if not (ok and frame is not None):
        cap.release()
        raise AssertionError("No se pudo leer frame inicial")

    print(f"Cámara {used_idx} OK — frame shape:", frame.shape)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Props reportadas -> W={w:.0f} H={h:.0f} FPS={fps:.0f}")

    means = []
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        means.append(float(frame.mean()))
        time.sleep(0.03)
    cap.release()

    print("Intensidades promedio (5 frames):", [round(m, 2) for m in means])
    if len(means) < 2:
        raise AssertionError("No se capturaron suficientes frames")

def test_03_image_io_and_basic_ops():
    _log("I/O de imagen y operaciones básicas")
    H, W = 480, 640
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    grad = (0.7 * x + 0.3 * y)
    img = np.dstack([
        (grad*255).astype(np.uint8),
        (np.flipud(grad)*255).astype(np.uint8),
        ((1.0 - grad)*255).astype(np.uint8),
    ])

    cv2.circle(img, (W//2, H//2), 60, (0, 255, 0), 3)
    cv2.rectangle(img, (50, 40), (180, 140), (255, 0, 0), 2)
    cv2.putText(img, "OpenCV Test", (50, H-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_img.png")
        ok_write = cv2.imwrite(path, img)
        if not (ok_write and os.path.exists(path)):
            raise AssertionError("Fallo al escribir imagen")
        img2 = cv2.imread(path, cv2.IMREAD_COLOR)
        if img2 is None or img2.shape != img.shape:
            raise AssertionError("Lectura de imagen inesperada")

    small = cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_AREA)
    if not (small.shape[0] == H//2 and small.shape[1] == W//2):
        raise AssertionError("Resize falló")

    M = cv2.getRotationMatrix2D((W/2, H/2), 30, 1.0)
    rot = cv2.warpAffine(img, M, (W, H))
    if rot.shape != img.shape:
        raise AssertionError("Rotación falló")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 80, 160)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contornos detectados:", len(cnts))
    if gray.ndim != 2 or edges.ndim != 2:
        raise AssertionError("Conversión de color o Canny no válidos")

def test_04_homography_and_warp():
    _log("Homografía y warpPerspective (sintético)")
    src = np.array([[100, 100], [540, 120], [520, 380], [120, 360]], dtype=np.float32)
    dst = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)

    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hm is None:
        raise AssertionError("No se pudo calcular la homografía")

    canvas = np.zeros((480, 640, 3), np.uint8)
    cv2.polylines(canvas, [src.astype(np.int32)], True, (0, 255, 0), 2)

    warped = cv2.warpPerspective(canvas, Hm, (400, 400))
    nonzero = int(np.count_nonzero(warped))
    print("Pixeles no-cero en warp:", nonzero)
    if nonzero <= 0:
        raise AssertionError("Warp resultó vacío")

def test_05_videowriter_short_clip():
    _log("VideoWriter — genera clip corto MJPG")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    H, W = 240, 320
    with tempfile.TemporaryDirectory() as td:
        vid_path = os.path.join(td, "test_mjpg.avi")
        vw = cv2.VideoWriter(vid_path, fourcc, 15.0, (W, H))
        if not vw.isOpened():
            print("No se pudo abrir VideoWriter (codec MJPG no disponible). Se omite.")
            return

        for t in range(30):
            frame = np.zeros((H, W, 3), np.uint8)
            cx = int((W/2) + 60*math.cos(t*0.2))
            cy = int((H/2) + 40*math.sin(t*0.25))
            cv2.circle(frame, (cx, cy), 20, (0, 255, 255), -1)
            cv2.putText(frame, f"t={t}", (10, H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            vw.write(frame)
        vw.release()

        size = os.path.getsize(vid_path) if os.path.exists(vid_path) else 0
        print("Video generado:", vid_path, "size:", size, "bytes")
        if size <= 0:
            raise AssertionError("VideoWriter no generó archivo")

# ---------------------------
# Runner sin pytest
# ---------------------------

SECTIONS = {
    "version": test_00_version_buildinfo,
    "backends": test_01_video_backends_and_cuda,
    "camera": test_02_camera_capture_if_available,
    "image_ops": test_03_image_io_and_basic_ops,
    "homography": test_04_homography_and_warp,
    "videowriter": test_05_videowriter_short_clip,
}

def run_selected(selected):
    failures = 0
    for name in selected:
        func = SECTIONS[name]
        try:
            func()
            print(f"✔ {name}: OK")
        except Exception as e:
            failures += 1
            print(f"✘ {name}: {e}")
    if failures:
        sys.exit(1)

def parse_args():
    ap = argparse.ArgumentParser(description="Runner de pruebas OpenCV (sin pytest)")
    ap.add_argument("--all", action="store_true", help="Ejecuta todas las secciones")
    ap.add_argument("--version", action="store_true", help="Versión y buildinfo")
    ap.add_argument("--backends", action="store_true", help="Backends de video y CUDA")
    ap.add_argument("--camera", action="store_true", help="Probar captura de cámara (usa CAM_INDEX si está)")
    ap.add_argument("--image-ops", action="store_true", help="I/O de imagen y operaciones básicas")
    ap.add_argument("--homography", action="store_true", help="Homografía y warpPerspective")
    ap.add_argument("--videowriter", action="store_true", help="Generar clip MJPG")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    to_run = []

    if args.all or not any([args.version, args.backends, args.camera, args.image_ops, args.homography, args.videowriter]):
        # Por defecto si no pasas flags: corre todo
        to_run = list(SECTIONS.keys())
    else:
        if args.version: to_run.append("version")
        if args.backends: to_run.append("backends")
        if args.camera: to_run.append("camera")
        if args.image_ops: to_run.append("image_ops")
        if args.homography: to_run.append("homography")
        if args.videowriter: to_run.append("videowriter")

    print("Secciones a ejecutar:", to_run)
    run_selected(to_run)
