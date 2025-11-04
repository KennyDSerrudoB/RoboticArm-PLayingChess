# prueba 1 de aruco
import cv2, io
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER  # 👈 carta
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ===== CONFIGURA =====
DICT       = cv2.aruco.DICT_5X5_100
IDS        = [10, 20, 30]     # genera varios por si acaso
SIDE_MM    = 35               # lado impreso del marcador (mm)
DPI        = 600              # 300–600 recomendado
MARGIN_MM  = 15               # márgenes del PDF
GAP_MM     = 12               # separación entre marcadores
# =====================

ppmm = DPI/25.4
PX = int(round(SIDE_MM * ppmm))  # pixeles por lado del propio marcador

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

c = canvas.Canvas("arucos_carta.pdf", pagesize=LETTER)
W, H = LETTER
x, y = MARGIN_MM*mm, H - MARGIN_MM*mm
gap  = GAP_MM*mm

cells   = 5                     # 5x5 porque DICT_5X5_100
cell_px = PX // cells
qz_px   = cell_px               # quiet zone = 1 celda blanca

for ID in IDS:
    # genera el marcador (sin borde) y agrega quiet zone blanca
    img = cv2.aruco.generateImageMarker(aruco_dict, ID, PX)
    img = np.pad(img, ((qz_px,qz_px),(qz_px,qz_px)), constant_values=255)

    # pasa a PNG en memoria
    png = cv2.imencode(".png", img)[1].tobytes()
    ir  = ImageReader(io.BytesIO(png))

    total_mm = img.shape[0] / ppmm
    w = h = total_mm * mm

    # salto de línea si no entra
    if x + w > W - MARGIN_MM*mm:
        x  = MARGIN_MM*mm
        y -= (h + gap)
        if y - h < MARGIN_MM*mm:
            c.showPage()
            x, y = MARGIN_MM*mm, H - MARGIN_MM*mm

    c.drawImage(ir, x, y - h, w, h, preserveAspectRatio=True, mask='auto')
    c.setFont("Helvetica", 9)
    c.drawString(x, y - h - 10, f"DICT_5X5_100  ID={ID}  {SIDE_MM}mm  {DPI}dpi")
    x += w + gap

c.save()
print("PDF generado: arucos_carta.pdf")
