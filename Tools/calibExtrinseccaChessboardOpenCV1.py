# ===============================================================
# Calibración Extrínseca con ArUco y tablero de ajedrez (KD3D Robotics)
# ===============================================================

import os
import cv2
import numpy as np
import json

# =============== CONFIGURACIÓN ===============
INTRINSIC_PATH   = "calib_capturas/session_20251103_132813/calibracion_intrinsecos.npz"
OUTPUT_DIR       = "extrinseca"
OUTPUT_NAME      = "T_cam_board_A1"
CAM_ID           = "/dev/video2"   # O usa CAM_ID = 2 si no abre

ARUCO_DICT       = cv2.aruco.DICT_5X5_100
ARUCO_ID         = 10
MARKER_SIZE_MM   = 35.0            # Lado del cuadrado negro (sin borde blanco)
MARKER_CORNER    = "H1"            # Esquina donde está pegado el ArUco
PLACEMENT        = "outside_touching"  # ArUco fuera, tocando el marco del tablero
N_SQUARES        = 8
SQUARE_SIZE_MM   = 40.25           # Tamaño real de una casilla

# Corrección por si el ArUco está rotado en el plano (Z)
ARUCO_YAW_DEG    = 90              # Prueba 0 / 90 / -90 / 180
# =============================================


# ------- FUNCIONES AUXILIARES -------

def load_intrinsics(path):
    """Carga intrínsecos desde .npz o .json detectando nombres comunes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró archivo de intrínsecos: {path}")

    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        K = np.array(data.get("cameraMatrix") or data.get("K") or data.get("mtx"))
        D = np.array(data.get("distCoeffs") or data.get("D") or data.get("dist"))
        if K is None or D is None:
            raise KeyError("No encontré 'cameraMatrix/K/mtx' o 'distCoeffs/D/dist' en el JSON")
        return K, D

    intr = np.load(path)
    keys = intr.files
    candidates_K = ["cameraMatrix", "K", "mtx", "arr_0"]
    candidates_D = ["distCoeffs", "D", "dist", "arr_1"]

    K = D = None
    for k in candidates_K:
        if k in keys:
            K = intr[k]
    for d in candidates_D:
        if d in keys:
            D = intr[d]

    if K is None or D is None:
        raise KeyError(f"No encontré K/D en {path}. Llaves disponibles: {keys}")
    return K, D


def T_from_rt(rvec, tvec):
    """Convierte rvec,tvec (OpenCV) a matriz homogénea 4x4."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = tvec.reshape(3,)
    return T


def Rz_deg(deg):
    """Rotación en Z (para corregir orientación del ArUco en el plano del tablero)."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def aruco_to_board_origin_offset(marker_corner: str,
                                 S_mm: float,
                                 N: int,
                                 m_mm: float,
                                 placement: str = "outside_touching") -> np.ndarray:
    """Vector (en marco del ArUco) desde su CENTRO hasta el ORIGEN del tablero (A1)."""
    W = (N - 1) * S_mm
    H = (N - 1) * S_mm
    c = marker_corner.upper()

    # --- vector del centro del ArUco a la esquina adyacente del tablero ---
    if placement == "outside_touching":
        if c == "A1":   local_corner = np.array([+m_mm/2, +m_mm/2, 0.0], dtype=float)
        elif c == "H1": local_corner = np.array([-m_mm/2, +m_mm/2, 0.0], dtype=float)
        elif c == "A8": local_corner = np.array([+m_mm/2, -m_mm/2, 0.0], dtype=float)
        elif c == "H8": local_corner = np.array([-m_mm/2, -m_mm/2, 0.0], dtype=float)
        else: raise ValueError("marker_corner debe ser A1/H1/A8/H8")
    elif placement == "on_square_corner":
        local_corner = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        raise ValueError("placement no reconocido")

    # --- traslación desde esa esquina local hasta A1 ---
    if c == "A1":   shift_to_A1 = np.array([ 0.0,  0.0, 0.0], dtype=float)
    elif c == "H1": shift_to_A1 = np.array([-W ,  0.0, 0.0], dtype=float)
    elif c == "A8": shift_to_A1 = np.array([ 0.0, -H , 0.0], dtype=float)
    elif c == "H8": shift_to_A1 = np.array([-W , -H , 0.0], dtype=float)

    return local_corner + shift_to_A1


# ------- PROGRAMA PRINCIPAL -------

def main():
    K, dist = load_intrinsics(INTRINSIC_PATH)
    print("\n[INFO] Matriz intrínseca K:\n", np.round(K, 3))
    print("[INFO] Distorsión:\n", np.round(dist, 5))

    ar_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params  = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ar_dict, params)

    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {CAM_ID}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n[INFO] Presiona 's' para guardar extrínseca | 'q' para salir\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids_flat = ids.flatten()
            for i, mid in enumerate(ids_flat):
                if mid == ARUCO_ID:
                    # Pose del ArUco
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], MARKER_SIZE_MM, K, dist
                    )
                    rvec = rvec[0][0]
                    tvec = tvec[0][0]

                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE_MM * 0.8)

                    cv2.putText(frame, f"tvec(mm): {np.round(tvec,1)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"rvec: {np.round(rvec,3)}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # --- Matrices ---
                    T_cam_aruco = T_from_rt(rvec, tvec)
                    T_fix = Rz_deg(ARUCO_YAW_DEG)
                    d = aruco_to_board_origin_offset(
                        marker_corner=MARKER_CORNER,
                        S_mm=SQUARE_SIZE_MM, N=N_SQUARES,
                        m_mm=MARKER_SIZE_MM, placement=PLACEMENT
                    )
                    T_aruco_to_A1 = np.eye(4)
                    T_aruco_to_A1[:3, 3] = d

                    # --- Extrínseca cámara -> tablero (A1) ---
                    T_cam_board_A1 = T_cam_aruco @ T_fix @ T_aruco_to_A1

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.npz")
                        np.savez(
                            out_path,
                            cameraMatrix=K,
                            distCoeffs=dist,
                            aruco_dict=str(ARUCO_DICT),
                            aruco_id=ARUCO_ID,
                            marker_size_mm=MARKER_SIZE_MM,
                            square_size_mm=SQUARE_SIZE_MM,
                            marker_corner=MARKER_CORNER,
                            placement=PLACEMENT,
                            yaw_fix_deg=ARUCO_YAW_DEG,
                            rvec=rvec, tvec=tvec,
                            T_cam_aruco=T_cam_aruco,
                            T_cam_board_A1=T_cam_board_A1
                        )
                        print("\n✅ [GUARDADO] Extrínseca en:", out_path)
                        print("T_cam_board_A1 =\n", np.round(T_cam_board_A1, 3))
                        cap.release()
                        cv2.destroyAllWindows()
                        return
        else:
            cv2.putText(frame, "Buscando ArUco...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Extrinseca ArUco (presiona 's' para guardar)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
