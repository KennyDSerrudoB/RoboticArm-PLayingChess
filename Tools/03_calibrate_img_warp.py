import cv2
import numpy as np
import glob
import os
import json

# ================================================================
# CONFIGURACIÓN FIJA
# ================================================================
# Ruta a la carpeta con tus capturas
FOLDER = "calib_capturas/session_20251103_132813"  # 🔧 cambia por tu carpeta exacta
# Patrón de esquinas internas (para tablero 8x8 cuadros)
CHESSBOARD_SIZE = (7, 7)
# Tamaño real de cada casilla en milímetros
SQUARE_SIZE_MM = 40.25
# ================================================================


def calibrar_intrinsecos():
    pattern = CHESSBOARD_SIZE
    square = SQUARE_SIZE_MM

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square

    objpoints, imgpoints, imgs_used = [], [], []
    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    # Leer todas las imágenes
    images = sorted(
        glob.glob(os.path.join(FOLDER, "*.png")) +
        glob.glob(os.path.join(FOLDER, "*.jpg"))
    )

    print(f"🔍 Buscando patrón {pattern} en {len(images)} imágenes de {FOLDER}…")

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ok, corners = cv2.findChessboardCorners(gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ok:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            imgs_used.append(path)
            print(f"  ✔ Detectado: {os.path.basename(path)}")
        else:
            print(f"  ⚠ No detectado: {os.path.basename(path)}")

    if len(imgpoints) < 10:
        print("❌ No hay suficientes imágenes válidas para calibrar (mínimo 10).")
        return

    print("\n📏 Calibrando cámara…")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Calcular error de reproyección
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print("\n==== RESULTADOS ====")
    print(f"📸 RMS total: {ret:.4f}")
    print(f"📉 Error medio: {mean_error:.4f} px")
    print("\nMatriz intrínseca (K):\n", K)
    print("\nCoeficientes de distorsión:\n", dist.ravel())

    # Guardar resultados
    np.savez(os.path.join(FOLDER, "calibracion_intrinsecos.npz"),
             K=K, dist=dist, rms=ret, error=mean_error, img_size=np.array(img_size))
    with open(os.path.join(FOLDER, "calibracion_intrinsecos.json"), "w") as f:
        json.dump({
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "rms": float(ret),
            "error_medio_px": float(mean_error),
            "img_size": img_size,
            "imagenes_usadas": imgs_used
        }, f, indent=2)

    print(f"\n💾 Guardado en: {FOLDER}/calibracion_intrinsecos.npz y .json")

    # Mostrar corrección visual
    print("\n🧪 Mostrando corrección (q para salir)…")
    for img_path in imgs_used[:3]:  # muestra 3 aleatorias
        img = cv2.imread(img_path)
        und = cv2.undistort(img, K, dist)
        combo = np.hstack([img, und])
        cv2.imshow("Original (izq)  |  Corregida (der)", combo)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrar_intrinsecos()
