# verificadorExtrinseca_auto_origen.py
import os, cv2, numpy as np

EXTR_PATH = "extrinseca/T_cam_board_H1.npz"  # o el que guardaste
CAM_ID    = "/dev/video2"
N         = 8

def npz_get(npz, keys, default=None):
    for k in keys:
        if k in npz.files:
            return npz[k]
    return default

def draw_axes(img, K, D, T, scale=50.0):
    pts = np.float32([[0,0,0],[scale,0,0],[0,scale,0],[0,0,scale]])
    R, t = T[:3,:3].astype(np.float32), T[:3,3].astype(np.float32)
    rvec, _ = cv2.Rodrigues(R)
    im, _ = cv2.projectPoints(pts, rvec, t, K, D)
    o,x,y,z = im.reshape(-1,2).astype(int)
    cv2.line(img, tuple(o), tuple(x), (0,0,255), 3)   # X rojo
    cv2.line(img, tuple(o), tuple(y), (0,255,0), 3)   # Y verde
    cv2.line(img, tuple(o), tuple(z), (255,0,0), 3)   # Z azul
    cv2.circle(img, tuple(o), 4, (0,255,255), -1)

def grid_pts_mm(S, N):
    return np.array([[x*S, y*S, 0.0] for y in range(N) for x in range(N)], np.float32)

def outline_mm(S, N):
    w = (N-1)*S
    return np.float32([[0,0,0],[w,0,0],[w,w,0],[0,w,0]])

def labels_mm(S, N, origin_corner="A1"):
    w = (N-1)*S
    # coords de las cuatro esquinas EN EL MARCO DEL ORIGEN
    # Si el origen es A1, A1=(0,0), H1=(w,0), A8=(0,w), H8=(w,w)
    # Si el origen es H1, H1=(0,0), A1=(w,0), H8=(0,w), A8=(w,w)
    if origin_corner == "A1":
        return {"A1":[0,0,0], "H1":[w,0,0], "A8":[0,w,0], "H8":[w,w,0]}
    if origin_corner == "H1":
        return {"H1":[0,0,0], "A1":[w,0,0], "H8":[0,w,0], "A8":[w,w,0]}
    if origin_corner == "A8":
        return {"A8":[0,0,0], "H8":[w,0,0], "A1":[0,w,0], "H1":[w,w,0]}
    if origin_corner == "H8":
        return {"H8":[0,0,0], "A8":[w,0,0], "H1":[0,w,0], "A1":[w,w,0]}
    return {"A1":[0,0,0],"H1":[w,0,0],"A8":[0,w,0],"H8":[w,w,0]}

def main():
    if not os.path.exists(EXTR_PATH):
        raise FileNotFoundError(EXTR_PATH)
    d = np.load(EXTR_PATH, allow_pickle=True)
    K = npz_get(d, ["cameraMatrix","K","mtx","arr_0"]).astype(np.float32)
    D = npz_get(d, ["distCoeffs","D","dist","arr_1"], np.zeros((5,),np.float32)).astype(np.float32)
    T = npz_get(d, ["T_cam_board_A1","T_cam_board_H1"])  # soporte ambos nombres
    if T is None:
        raise KeyError("No encontré T_cam_board_* en el archivo.")
    S = float(npz_get(d, ["square_size_mm"], 40.25))
    origin_corner = str(npz_get(d, ["marker_corner"], "A1"))

    pts   = grid_pts_mm(S, N)
    out3d = outline_mm(S, N)
    lab3d = labels_mm(S, N, origin_corner)

    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {CAM_ID}")

    print(f"[i] Origen en: {origin_corner} | q/ESC para salir")
    while True:
        ok, img = cap.read()
        if not ok: break
        R, t = T[:3,:3].astype(np.float32), T[:3,3].astype(np.float32)
        rvec, _ = cv2.Rodrigues(R)

        impts, _ = cv2.projectPoints(pts, rvec, t, K, D)
        for p in impts.reshape(-1,2):
            cv2.circle(img, tuple(np.int32(p)), 3, (0,255,0), -1)

        poly, _ = cv2.projectPoints(out3d, rvec, t, K, D)
        cv2.polylines(img, [poly.reshape(-1,2).astype(int)], True, (255,255,0), 2)

        draw_axes(img, K, D, T, scale=S*1.5)

        for name, P in lab3d.items():
            P = np.float32([P])
            P2, _ = cv2.projectPoints(P, rvec, t, K, D)
            p = tuple(np.int32(P2.reshape(-1,2)[0]))
            cv2.putText(img, name, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Verificacion Extrinseca", img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
