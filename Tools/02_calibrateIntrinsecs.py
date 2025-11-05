# Leemos el dataset y creamos la calibracion
# hallamos el parametro K
import cv2 as cv
import numpy as np
import json, csv, os

# ====== CONFIGURACIÓN ======
IMG_PATH = "calib_capturas/tablero.jpg"   # Ruta de imagen del tablero (se tomará de cámara si no existe)
CAMERA_ID = "/dev/video2"               # ID de cámara (0 = por defecto)
WARP_SIZE = 800            # px (800x800 → 100 px por casilla)
CELL = WARP_SIZE // 8

# Nombres de columnas
FILES = ['a','b','c','d','e','f','g','h']

# Coordenadas robot de las 4 esquinas TL,TR,BR,BL (mm)
ROBOT_TL = None  # Ejemplo: (100.0, 500.0)
ROBOT_TR = None
ROBOT_BR = None
ROBOT_BL = None

Z_TABLERO = 0.0         # mm: altura del tablero
Z_APPROACH = Z_TABLERO + 50.0
PINZA_OFFSET_Z = 0.0

# ====== FUNCIONES ======
def ask_robot_points_if_needed():
    global ROBOT_TL, ROBOT_TR, ROBOT_BR, ROBOT_BL
    if None in (ROBOT_TL, ROBOT_TR, ROBOT_BR, ROBOT_BL):
        print("\nIntroduce 4 puntos ROBOT (mm) en orden TL,TR,BR,BL. Ej: 100 500")
        def read_xy(msg):
            nums = input(msg).strip().split()
            return (float(nums[0]), float(nums[1]))
        ROBOT_TL = read_xy("Robot TL (X Y): ")
        ROBOT_TR = read_xy("Robot TR (X Y): ")
        ROBOT_BR = read_xy("Robot BR (X Y): ")
        ROBOT_BL = read_xy("Robot BL (X Y): ")

def compute_homography(src_pts, dst_pts):
    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    H = cv.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)
    return H, H_inv

def map_points(H, pts):
    pts = np.float32(pts).reshape(-1,1,2)
    out = cv.perspectiveTransform(pts, H).reshape(-1,2)
    return [(float(x), float(y)) for x,y in out]

def grid_centers():
    centers = []
    for r in range(8):
        for c in range(8):
            cx = int(c*CELL + CELL/2)
            cy = int((7-r)*CELL + CELL/2)  # r=0 => fila 1 abajo
            name = f"{FILES[c]}{r+1}"
            centers.append(((cx,cy), name))
    return centers

# ====== CAPTURA O CARGA DE IMAGEN ======
def get_board_image():
    if os.path.exists(IMG_PATH):
        print(f"📂 Usando imagen existente: {IMG_PATH}")
        orig = cv.imread(IMG_PATH)
        if orig is None:
            raise RuntimeError("⚠️ No pude leer la imagen. Archivo dañado o ruta incorrecta.")
        return orig

    print("📸 No se encontró tablero.jpg, capturando imagen de la cámara...")
    cap = cv.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("No pude acceder a la cámara.")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("No pude capturar imagen del tablero.")
    cv.imwrite(IMG_PATH, frame)
    print(f"✅ Imagen capturada y guardada como {IMG_PATH}")
    return frame

# ====== INTERFAZ DE USUARIO ======
clicked = []

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))
        print(f"Clic {len(clicked)} en: {x}, {y}")

def show_help():
    print("\n🧭 INSTRUCCIONES:")
    print("1️⃣ Haz 4 clics en las esquinas del tablero en este orden:")
    print("   TL (arriba-izquierda), TR (arriba-derecha), BR (abajo-derecha), BL (abajo-izquierda)")
    print("2️⃣ Presiona 'r' para reiniciar los clics.")
    print("3️⃣ Presiona 'q' para continuar cuando tengas los 4 puntos.\n")

# ====== FUNCIÓN PRINCIPAL ======
def main():
    orig = get_board_image()
    show_help()
    win = "Clic 4 esquinas (TL->TR->BR->BL)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, on_mouse)

    while True:
        draw = orig.copy()
        for i, (x,y) in enumerate(clicked):
            cv.circle(draw, (x,y), 6, (0,255,0), -1)
            cv.putText(draw, f"{i+1}", (x+6,y-6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv.imshow(win, draw)
        key = cv.waitKey(20) & 0xFF
        if key == ord('r'):
            clicked.clear()
            print("↩️ Reiniciado.")
        elif key == ord('q'):
            if len(clicked) == 4:
                break
            else:
                print("⚠️ Aún no tienes 4 puntos.")

    cv.destroyWindow(win)
    corners_img = np.float32(clicked)

    # Homografía IMG->WARP
    dst_corners = np.float32([[0,0],[WARP_SIZE,0],[WARP_SIZE,WARP_SIZE],[0,WARP_SIZE]])
    H_img2warp, _ = compute_homography(corners_img, dst_corners)
    warp = cv.warpPerspective(orig, H_img2warp, (WARP_SIZE, WARP_SIZE))

    # Mostrar warp con centros
    preview = warp.copy()
    centers = grid_centers()
    for (cx,cy), name in centers:
        cv.circle(preview, (cx,cy), 4, (0,255,0), -1)
        cv.putText(preview, name, (cx-12, cy-6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv.LINE_AA)

    cv.imshow("Vista cenital (warp) + centros", preview)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Guardar datos
    data = [{"square": name, "cx_warp": cx, "cy_warp": cy}
            for (cx,cy), name in centers]
    with open("centros_warp.json", "w") as f:
        json.dump({"WARP_SIZE": WARP_SIZE, "centers": data}, f, indent=2)
    print("💾 Guardado centros_warp.json")

    # (Opcional) Warp -> Robot
    use_robot = input("¿Mapear a coordenadas ROBOT? (s/n): ").strip().lower() == 's'
    if use_robot:
        ask_robot_points_if_needed()
        src = np.float32([[0,0],[WARP_SIZE,0],[WARP_SIZE,WARP_SIZE],[0,WARP_SIZE]])
        dst = np.float32([ROBOT_TL,ROBOT_TR,ROBOT_BR,ROBOT_BL])
        H_warp2robot, _ = compute_homography(src, dst)

        pts_warp = [(d["cx_warp"], d["cy_warp"]) for d in data]
        pts_robot = map_points(H_warp2robot, pts_warp)

        rows = []
        for (d, (xr,yr)) in zip(data, pts_robot):
            rows.append({
                "square": d["square"],
                "X": xr,
                "Y": yr,
                "Z": Z_TABLERO + PINZA_OFFSET_Z,
                "Z_approach": Z_APPROACH
            })

        with open("casillas_robot.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["square","X","Y","Z","Z_approach"])
            w.writeheader()
            w.writerows(rows)

        np.save("H_warp2robot.npy", H_warp2robot)
        np.save("H_img2warp.npy", H_img2warp)
        print("✅ Guardados: casillas_robot.csv, H_warp2robot.npy, H_img2warp.npy")

# ====== EJECUCIÓN ======
if __name__ == "__main__":
    main()
