# Pc/vision/register_board.py
"""
Registro de tablero de ajedrez:
- Detecta el rectángulo del tablero (auto o manual).
- Calcula homografía y genera una vista "warp" (cenital) 800x800 px.
- Aplica des-distorsión (intrínsecos), refinamiento con patrón 7×7,
  y permite compensar bordes con inset por lado (top/right/bottom/left).
- Etiqueta casillas a1..h8 y guarda:
  H_img2warp.npy, empty_warp.png, casillas_pixels.csv
"""

from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np
import cv2

# permitir importar paquetes del repo si se corre como script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Pc.common.config_loader import load_config, get_path
from Pc.vision.cam_io import open_camera_from_config


# =========================
#  Utils geométricas / Warp
# =========================
def _order_pts_clockwise(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos en sentido horario y devuelve [tl, tr, br, bl]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _find_board_quad_auto(img: np.ndarray) -> np.ndarray | None:
    """Encuentra el cuadrilátero más grande que parezca el tablero."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    h, w = img.shape[:2]
    area_img = w * h
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.15 * area_img:
                return _order_pts_clockwise(approx.reshape(4, 2))
    return None

def _warp(img: np.ndarray, src4: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp a cuadrado size×size. Devuelve (warp, H_img2warp)."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src4.astype(np.float32), dst)
    warp = cv2.warpPerspective(img, H, (size, size))
    return warp, H

def _inset_quad(pts: np.ndarray, inset_px: float) -> np.ndarray:
    """Inset uniforme hacia el centro."""
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    v = c - pts
    n = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    return (pts + n * inset_px).astype(np.float32)

def _inset_quad_asymmetric(pts: np.ndarray, inset: dict[str, float]) -> np.ndarray:
    """
    Inset por lado: top/right/bottom/left.
    Mueve cada borde hacia el centro según su valor.
    Orden de pts esperado: [tl, tr, br, bl].
    """
    pts = pts.astype(np.float32)
    tl, tr, br, bl = pts
    c = pts.mean(axis=0)

    def move_edge(p1, p2, amount):
        if not amount:
            return p1, p2
        mid = (p1 + p2) * 0.5
        n = c - mid
        n = n / (np.linalg.norm(n) + 1e-6)
        return p1 + n * amount, p2 + n * amount

    tl, tr = move_edge(tl, tr, float(inset.get("top", 0)))
    tr, br = move_edge(tr, br, float(inset.get("right", 0)))
    br, bl = move_edge(br, bl, float(inset.get("bottom", 0)))
    bl, tl = move_edge(bl, tl, float(inset.get("left", 0)))
    return np.array([tl, tr, br, bl], np.float32)


# =====================
#  Etiquetas y dibujo
# =====================
def _labels_grid() -> np.ndarray:
    """Grilla fija: TL=a1, TR=a8, BL=h1, BR=h8 (convención acordada)."""
    files = list("abcdefgh")
    grid = np.empty((8, 8), dtype=object)
    for r in range(8):
        for c in range(8):
            grid[r, c] = f"{files[r]}{c+1}"
    return grid

def _draw_grid_labels(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    cell = w // 8
    for i in range(1, 8):
        cv2.line(vis, (i * cell, 0), (i * cell, h), (0, 255, 0), 1)
        cv2.line(vis, (0, i * cell), (w, i * cell), (0, 255, 0), 1)
    for r in range(8):
        for c in range(8):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)
            cv2.putText(
                vis, labels[r, c], (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )
    return vis


# =====================
#  Input manual (Clicks)
# =====================
_clicks: list[tuple[int, int]] = []

def _on_mouse(event, x, y, flags, param):
    global _clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))

def _get_corners_manual(img: np.ndarray) -> np.ndarray | None:
    """Clic en 4 esquinas del tablero en orden TL, TR, BR, BL."""
    global _clicks
    _clicks = []
    tmp = img.copy()
    cv2.namedWindow("Seleccion manual")
    cv2.setMouseCallback("Seleccion manual", _on_mouse)
    print("👉 Clic: TL, TR, BR, BL.  [ENTER]=OK, [r]=reinicia, [q]=cancela")

    while True:
        disp = tmp.copy()
        for i, (x, y) in enumerate(_clicks):
            cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(disp, f"{i+1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Seleccion manual", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(_clicks) == 4:  # ENTER
            break
        elif k in (ord('r'), ord('R')):
            _clicks = []
        elif k in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow("Seleccion manual")
            return None

    cv2.destroyWindow("Seleccion manual")
    return np.array(_clicks, np.float32)


# =====================
#  Cámara y refinamiento
# =====================
def _load_intrinsics(npz_path: Path):
    if not npz_path.exists():
        return None, None
    data = np.load(str(npz_path))
    K = data.get("K") or data.get("camera_matrix")
    dist = data.get("dist") or data.get("dist_coeffs")
    if K is None or dist is None:
        return None, None
    return K.astype(np.float32), dist.astype(np.float32)

def _undistort(img: np.ndarray, K, dist) -> np.ndarray:
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0)
    return cv2.undistort(img, K, dist, None, newK)

def _refine_on_warp(warp: np.ndarray) -> np.ndarray | None:
    """Alinea con patrón interno 7×7 para corregir pequeñas desviaciones."""
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE
    )
    if not ok:
        return None
    corners = corners.squeeze(1).astype(np.float32)
    size = warp.shape[1]
    cell = size / 8.0
    dst = np.array([[ (i+1)*cell, (j+1)*cell ] for j in range(7) for i in range(7)], np.float32)
    Hcorr, _ = cv2.findHomography(corners, dst, cv2.RANSAC, 3.0)
    if Hcorr is None:
        return None
    return cv2.warpPerspective(warp, Hcorr, (size, size))


# =============
#  Principal
# =============
def main():
    cfg = load_config()

    paths = cfg.get("paths", {})
    intr_npz = Path(paths.get("intrinsics_file", "Data/camera_intrinsics.npz"))
    use_undist = bool(cfg.get("camera", {}).get("undistort", True))
    K, dist = _load_intrinsics(intr_npz)

    warp_size = int(cfg.get("board", {}).get("warp_size", 800))
    edge_inset_px = float(cfg.get("board", {}).get("edge_inset_px", 0))
    edge_inset = cfg.get("board", {}).get("edge_inset", {}) or {}

    H_path = Path(paths.get("homography_img2warp", get_path("data_dir") / "H_img2warp.npy"))
    empty_path = Path(paths.get("empty_warp", get_path("data_dir") / "empty_warp.png"))
    csv_path = Path(paths.get("squares_csv", get_path("data_dir") / "casillas_pixels.csv"))

    cam = open_camera_from_config()
    print("Teclas: [a]=auto  [m]=manual  [s]=guardar  [q]=salir")

    H = warp = src4 = None

    try:
        while True:
            ok, frame = cam.read(timeout_ms=int(cfg.get("camera", {}).get("read_timeout_ms", 1000)))
            if ok and use_undist and K is not None:
                frame = _undistort(frame, K, dist)

            if not ok:
                blank = np.zeros((240, 320, 3), np.uint8)
                cv2.putText(blank, "No frame", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("RegistroTablero", blank)
                continue

            show = frame.copy()
            if src4 is not None:
                for i, (x, y) in enumerate(src4.astype(int)):
                    cv2.circle(show, (x, y), 6, (0, 255, 255), -1)
                    cv2.putText(show, f"{i+1}", (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("RegistroTablero", show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            elif k == ord('a'):
                quad = _find_board_quad_auto(frame)
                if quad is None:
                    print("❌ Auto: no se encontró cuadrilátero convincente.")
                    continue

                src4 = quad
                src4_use = src4
                if edge_inset:
                    src4_use = _inset_quad_asymmetric(src4_use, edge_inset)
                elif edge_inset_px:
                    src4_use = _inset_quad(src4_use, edge_inset_px)

                warp, H = _warp(frame, src4_use, warp_size)
                warp2 = _refine_on_warp(warp)
                if warp2 is not None:
                    warp = warp2

                print("✅ Auto: tablero detectado.")
                vis = _draw_grid_labels(warp, _labels_grid())
                cv2.imshow("Warp", vis)

            elif k == ord('m'):
                pts = _get_corners_manual(frame)
                if pts is None:
                    continue

                src4 = _order_pts_clockwise(pts)
                src4_use = src4
                if edge_inset:
                    src4_use = _inset_quad_asymmetric(src4_use, edge_inset)
                elif edge_inset_px:
                    src4_use = _inset_quad(src4_use, edge_inset_px)

                warp, H = _warp(frame, src4_use, warp_size)
                warp2 = _refine_on_warp(warp)
                if warp2 is not None:
                    warp = warp2

                print("✅ Manual: homografía calculada.")
                vis = _draw_grid_labels(warp, _labels_grid())
                cv2.imshow("Warp", vis)

            elif k == ord('s'):
                if H is None or warp is None:
                    print("⚠️ Nada que guardar. Usa [a] o [m] primero.")
                    continue

                # Guardar homografía e imagen "vacía" warp
                H_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(H_path), H)
                cv2.imwrite(str(empty_path), warp)

                # Guardar centros y etiquetas
                cell = warp_size // 8
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["square", "cx_px", "cy_px"])
                    labels = _labels_grid()
                    for r in range(8):
                        for c in range(8):
                            cx = int((c + 0.5) * cell)
                            cy = int((r + 0.5) * cell)
                            w.writerow([labels[r, c], cx, cy])

                print(f"💾 Guardado:\n  H -> {H_path}\n  empty_warp -> {empty_path}\n  squares -> {csv_path}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Cámara liberada.")


if __name__ == "__main__":
    main()
