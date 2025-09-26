# alert_if_3_faces.py
# Requisitos: opencv-python, numpy
import cv2
import time
import platform
import sys
import os

# Beep en Windows con winsound; fallback a campana ASCII
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except Exception:
        WINSOUND_AVAILABLE = False
else:
    WINSOUND_AVAILABLE = False

def play_beep(duration_ms=220, freq=800):
    if WINSOUND_AVAILABLE:
        try:
            winsound.Beep(freq, duration_ms)
            return
        except Exception:
            pass
    sys.stdout.write("\a")
    sys.stdout.flush()

def detect_faces(frame, face_cascade, scale=0.75):
    """Devuelve lista de rects de caras en coordenadas del frame original."""
    # Redimensionar para acelerar y normalizar detección
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Mejorar contraste localmente
    gray = cv2.equalizeHist(gray)

    # detectMultiScale tuning: ajustá scaleFactor y minNeighbors si hace falta
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    # Escalar rects al tamaño original
    rects = []
    for (x, y, w, h) in faces:
        rects.append((int(x/scale), int(y/scale), int(w/scale), int(h/scale)))
    return rects

def main():
    # Ruta al xml incluido en la instalación de OpenCV
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        print("No encuentro el archivo cascade:", cascade_path)
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error cargando el clasificador Haar.")
        return

    cap = cv2.VideoCapture(0)  # usar cámara por defecto
    if not cap.isOpened():
        print("No puedo abrir la cámara")
        return

    cooldown_s = 5.0      # tiempo mínimo entre alertas
    last_alert_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rects = detect_faces(frame, face_cascade, scale=0.6)

        # Opcional: filtrar detecciones muy pequeñas
        faces_filtered = []
        for (x,y,w,h) in rects:
            if w*h < 1500:   # umbral: ajustalo según cámara/resize
                continue
            faces_filtered.append((x,y,w,h))

        count_faces = len(faces_filtered)

        # Dibujar rectángulos y etiqueta
        for (x,y,w,h) in faces_filtered:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,200,0), 2)
            cv2.putText(frame, "Cara", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)

        # Contador en pantalla
        cv2.putText(frame, f"Caras: {count_faces}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        # Lógica de alerta
        if count_faces >= 3:
            now = time.time()
            if now - last_alert_time >= cooldown_s:
                last_alert_time = now
                cv2.putText(frame, "!!! ALERTA: 3+ CARAS !!!", (10,80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 3, cv2.LINE_AA)
                play_beep()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALERTA: {count_faces} caras detectadas")

        cv2.imshow("Deteccion de caras - q para salir", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
