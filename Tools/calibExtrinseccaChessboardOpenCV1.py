#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimación de extrínsecos (pose tablero -> cámara) con OpenCV
- Usa tus intrínsecos (K, dist) ya calibrados
- Detecta checkerboard (7x7 para tablero 8x8)
- PnP robusto: solvePnPRansac + solvePnPRefineLM
- Promedio móvil para estabilizar
- Dibuja ejes XYZ y guarda T, quat, rpy en .json y .npz

Controles:
  s -> guardar extrínsecos
  c -> limpiar promedio
  q -> salir
"""

import cv2
import numpy as np
import json
import os
from collections import deque
from datetime import datetime
from math import atan2, asin

# =======================
# ====== CONFIG =========
# =======================
CAMERA_ID   = "/dev/video2" 
FRAME_SIZE  = (1280, 720)            # usa la misma resolución que los intrínsecos
# Carpeta de tu sesión de intrínsecos (donde está calibracion_intrinsecos.npz)
FOLDER_INTR = "calib_capturas/session_20251103_132813"
INTR_FILE   = os.path.join(FOLDER_INTR, "calibracion_intrinsecos.npz")

# Checkerboard de un tablero 8x8 -> 7x7 esquinas internas
CHESSBOARD_SIZE = (7, 7)             # (cols, rows) = (horiz, vert)
SQUARE_MM       = 25.0               # tamaño de casilla (mm)

AXIS_LEN_MM = 60                     # longitud de ejes dibujados
AVG_WINDOW  = 15                     # frames para promedio móvil
SAVE_DIR    = FOLDER_INTR            # dónde guardar extrínsecos calculados
# =======================


# ---------- Utilidades geométricas ----------
def load_intrinsics(path_npz: str):
    data = np.load(path_npz)
    K = data["K"]
    dist = data["dist"]
    img_size = tuple(data["img_size"])
    return K, dist, img_size

def chessboard_object_points(pattern, square_mm):
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    # Plano tablero: Z=0, X->derecha del patrón, Y->abajo del patrón
    return objp

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = tvec.reshape(3)
    return T

def invert_T(T):
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def R_to_quat(R):
    # Hamilton (w, x, y, z)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)

def R_to_rpy(R):
    # Roll-Pitch-Yaw (rad), convención XYZ intrínseca
    sy = -R[2,0]
    pitch = asin(np.clip(sy, -1.0, 1.0))
    roll  = atan2(R[2,1], R[2,2])
    yaw   = atan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

def draw_axes(img, K, dist, rvec, tvec, axis_len):
    axes = np.float32([
        [0,0,0],
        [axis_len,0,0],
        [0,axis_len,0],
        [0,0,axis_len]
    ])
    pts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
    p0 = tuple(pts[0].ravel().astype(int))
    px = tuple(pts[1].ravel().astype(int))
    py = tuple(pts[2].ravel().astype(int))
    pz = tuple(pts[3].ravel().astype(int))
    cv2.line(img, p0, px, (0,0,255), 3)   # X rojo
    cv2.line(img, p0, py, (0,255,0), 3)   # Y verde
    cv2.line(img, p0, pz, (255,0,0), 3)   # Z azul
    return img

def overlay_text(img, txt, y=30, color=(0,255,0)):
    cv2.putText(img, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def save_pose(T_cam_from_board, K, dist, used_size):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(SAVE_DIR, f"extrinsecos_{ts}.json")
    out_npz  = os.path.join(SAVE_DIR, f"extrinsecos_{ts}.npz")

    T_board_from_cam = invert_T(T_cam_from_board)
    R = T_cam_from_board[:3,:3]
    t = T_cam_from_board[:3, 3]
    quat = R_to_quat(R)   # (w,x,y,z)
    rpy  = R_to_rpy(R)    # rad

    payload = {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "image_size": used_size,
        "T_cam_from_board": T_cam_from_board.tolist(),
        "T_board_from_cam": T_board_from_cam.tolist(),
        "tvec_cam_from_board_mm": t.tolist(),
        "quat_cam_from_board_wxyz": quat.tolist(),
        "rpy_cam_from_board_rad": rpy.tolist(),
        "notes": "Ejes tablero: X->derecha del patrón, Y->abajo del patrón, Z->hacia la cámara (mano derecha)."
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    np.savez(out_npz, **{k:np.array(v) if isinstance(v,list) else v for k,v in payload.items()})
    print(f"💾 Guardado:\n  {out_json}\n  {out_npz}")


# ------------- Main -------------
def main():
    # Intrínsecos
    K, dist, img_size = load_intrinsics(INTR_FILE)
    print("K:\n", K, "\ndist:\n", dist.ravel(), "\nimg_size:", img_size)

    # Objetos 3D del tablero (mm)
    objp = chessboard_object_points(CHESSBOARD_SIZE, SQUARE_MM)

    # Cámara
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    # buffers de promedio
    rque = deque(maxlen=AVG_WINDOW)
    tque = deque(maxlen=AVG_WINDOW)

    print("Controles: s=guardar extrínsecos | c=limpiar promedio | q=salir")

    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, cb_flags)
        vis = frame.copy()

        if found:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria_subpix)

            # ---- PnP robusto (muchos puntos) ----
            okpnp, rvec, tvec, inliers = cv2.solvePnPRansac(
                objp, corners, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=1.5,
                confidence=0.999,
                iterationsCount=200
            )

            if okpnp and inliers is not None and len(inliers) >= 6:
                # Refinar con inliers
                rvec, tvec = cv2.solvePnPRefineLM(
                    objp[inliers[:,0]], corners[inliers[:,0]], K, dist, rvec, tvec
                )

                rque.append(rvec.reshape(3))
                tque.append(tvec.reshape(3))

                # promedio móvil
                rmean = np.mean(np.vstack(rque), axis=0).reshape(3,1)
                tmean = np.mean(np.vstack(tque), axis=0).reshape(3,1)

                # dibujar
                cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, True)
                vis = draw_axes(vis, K, dist, rmean, tmean, AXIS_LEN_MM)

                T_cb = rvec_tvec_to_T(rmean, tmean)
                R = T_cb[:3,:3]
                rpy = R_to_rpy(R)

                overlay_text(vis, f"Patrón: SÍ | z_mm={tmean[2,0]:7.1f} | inliers={len(inliers)}", 30, (0,255,0))
                overlay_text(vis, f"rpy(rad)=[{rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}]", 60, (255,255,255))

                cv2.imshow("Extrinsecos - Live", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    save_pose(T_cb, K, dist, img_size)
                elif key == ord('c'):
                    rque.clear(); tque.clear()
                elif key == ord('q'):
                    break
            else:
                overlay_text(vis, "PnP falló (pocos inliers)", 30, (0,0,255))
                cv2.imshow("Extrinsecos - Live", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        else:
            overlay_text(vis, "Patrón: NO", 30, (0,0,255))
            cv2.imshow("Extrinsecos - Live", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
