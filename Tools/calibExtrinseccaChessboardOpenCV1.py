import cv2
import numpy as np

# ========== CONFIGURACIÓN ==========
CAMERA_ID = "/dev/video2"    # cambia si usas otra cámara o archivo
WIDTH, HEIGHT = 1280, 720
OUTPUT_SIZE = 800  # tablero final en píxeles (800x800)
# ===================================

# Rango de colores HSV (ajustables según tu iluminación)
COLOR_RANGES = {
    "red":      ([0, 100, 100], [10, 255, 255]),        # 🔴 A1
    "green":    ([40, 60, 60], [80, 255, 255]),         # 🟢 H1
    "blue":     ([100, 100, 60], [140, 255, 255]),      # 🔵 H8
    "yellow":   ([20, 100, 100], [35, 255, 255])        # 🟡 A8
}

# Abrir cámara
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print("Presiona 'q' para salir cuando se muestre la vista cenital")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer la cámara.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_points = {}

    # --- Detectar cada color ---
    for name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_points[name] = (cx, cy)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                cv2.putText(frame, name, (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # --- Si se detectaron los 4 puntos ---
    if len(detected_points) == 4:
        red_pt = detected_points["red"]       # A1
        green_pt = detected_points["green"]   # H1
        blue_pt = detected_points["blue"]     # H8
        yellow_pt = detected_points["yellow"] # A8

        src_pts = np.float32([red_pt, green_pt, blue_pt, yellow_pt])
        dst_pts = np.float32([
            [0, OUTPUT_SIZE],            # A1 -> abajo izquierda
            [OUTPUT_SIZE, OUTPUT_SIZE],  # H1 -> abajo derecha
            [OUTPUT_SIZE, 0],            # H8 -> arriba derecha
            [0, 0]                       # A8 -> arriba izquierda
        ])

        H, _ = cv2.findHomography(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, H, (OUTPUT_SIZE, OUTPUT_SIZE))

        cv2.imshow("Vista cenital (homografía)", warped)

    cv2.imshow("Vista original con detección", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
