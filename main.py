
import time
import requests
import cv2
import numpy as np

from requests.auth import HTTPDigestAuth
from ultralytics import solutions


# -----------------------------------------
# CONFIGURACIÓN
# -----------------------------------------

URL_FRAME = "https://media.istockphoto.com/id/1457298174/es/foto/ciudadanos-cruzando-cebra-abarrotada-tiro-ancho.jpg?s=612x612&w=0&k=20&c=ZiERaYOfResMvzG3BkYS7LGLpI1aPbBCEWBmPc7wu84="
# URL_FRAME = "https://img.magnific.com/foto-gratis/tiro-completo-amigos-caminando-juntos_23-2148618878.jpg?semt=ais_hybrid&w=740&q=80"
# URL_FRAME = "https://media.istockphoto.com/id/1364054930/es/foto/multitud-de-personas-irreconocibles-cruzando-la-calle-en-el-sem%C3%A1foro-cebra.jpg?s=612x612&w=0&k=20&c=IfD3VDx9E0QVZB8UzFe84f47P9C9TCigkVwndmcGzJA="
# URL_FRAME = "http://192.168.6.188/ISAPI/Streaming/channels/101/picture"

DVR_USERNAME = "admin"
DVR_PASSWORD = "12345"

INTERVALO_SEGUNDOS = 3

# Coordenadas del polígono de interés (x, y)
# IMPORTANTE: adaptar a la resolución de tu imagen.
REGION = {
    "personas": [
        (0, 0),
        (700, 0),
        (700, 500),
        (0, 500)
    ]
}


# -----------------------------------------
# INICIALIZACIÓN DEL CONTADOR
# -----------------------------------------

contador = solutions.RegionCounter(
    model="yolo26n.pt",
    region=REGION,
    classes=[0],       # Solo personas
    conf=0.1, #0.25
    tracker="bytetrack.yaml",
    verbose=False,
    iou=0.95
)


# -----------------------------------------
# FUNCIÓN DE CONTEO
# -----------------------------------------

def contar_personas():

    try:
        # Descargar una imagen nueva
        r = requests.get(
            URL_FRAME,
            auth=HTTPDigestAuth(
                DVR_USERNAME,
                DVR_PASSWORD
            ),
            timeout=10
        )

        r.raise_for_status()

        # Convertir JPG recibido a imagen OpenCV
        imagen_bytes = np.frombuffer(r.content, np.uint8)
        imagen = cv2.imdecode(
            imagen_bytes,
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            raise ValueError("No se pudo decodificar la imagen JPG")

        # Ejecutar YOLO26 y analizar la región
        resultado = contador(imagen)

        # Obtener cantidad de personas dentro de la región
        cantidad = resultado.region_counts.get(
            "personas", 0
        )

        return int(cantidad)

    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None


# -----------------------------------------
# BUCLE PRINCIPAL
# -----------------------------------------

if __name__ == "__main__":

    while True:

        inicio = time.monotonic()

        cantidad = contar_personas()

        if cantidad is not None:
            print(f"Personas en el área: {cantidad}")
            requests.get(url="http://localhost/mi_api.php", params={"cant": cantidad})

        # Mantener intervalo de 3 segundos
        tiempo_procesamiento = time.monotonic() - inicio

        time.sleep(
            max(0, INTERVALO_SEGUNDOS - tiempo_procesamiento)
        )
