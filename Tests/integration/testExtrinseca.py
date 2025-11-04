import cv2
import numpy as np

# --- Paths ---
EXTR_PATH = "extrinseca/T_cam_board_A1.npz"
IMG_PATH  = "extrinseca/frame_prueba.png"  # o captura nueva

# --- Carga extrínseca ---
data = np.load(EXTR_PATH)
K, dist = data["cameraMatrix"], data["distCoeffs"]
rvec, tvec = data["rvec"], data["tvec"]

# --- Parámetros tablero ---
N = 8
SQUARE_SIZE_MM = data["square_size_mm"]
axis_points = []
for y in range(N):
    for x in range(N):
        axis_points.append([x * SQUARE_SIZE_MM, y * SQUARE_SIZE_MM, 0])
axis_points = np.array(axis_points, dtype=np.float32)

# --- Cargar imagen o capturar ---
img = cv2.imread(IMG_PATH)
if img is None:
    cap = cv2.VideoCapture(2)
    _, img = cap.read()
    cap.release()

# --- Proyectar puntos ---
imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)

# --- Dibujar ---
for pt in imgpts:
    x, y = int(pt[0][0]), int(pt[0][1])
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

cv2.imshow("Verificacion Extrinseca", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
