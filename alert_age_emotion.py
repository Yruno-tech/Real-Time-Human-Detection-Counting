# alert_age_emotion.py
# Muestra edad y emoción por cara, con suposición estable por cara (historial + moda)
# Requisitos: pip install opencv-python numpy (y tensorflow si usás emotion_model.hdf5)

import cv2
import os
import numpy as np
import time
from collections import Counter

# ---------------- Config ----------------
MODELS_DIR = "models"
# Ajustables
HISTORY_LEN = 10           # cuántas predicciones guardar por tracker (10 está bien)
DIST_THRESHOLD = 80        # distancia máxima para considerar que es la misma cara (px)
MAX_MISSED_FRAMES = 30     # después de tantos frames sin match, borramos el tracker

# Haar cascade
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Edad (intenta cargar varios nombres comunes)
AGE_PROTO_CANDIDATES = [os.path.join(MODELS_DIR, "deploy_age.prototxt"),
                       os.path.join(MODELS_DIR, "age_deploy.prototxt")]
AGE_MODEL_CANDIDATES = [os.path.join(MODELS_DIR, "age_net.caffemodel")]

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
            '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(54-59)', '(60-100)']

age_net = None
def try_load_age_net():
    for p in AGE_PROTO_CANDIDATES:
        for m in AGE_MODEL_CANDIDATES:
            if os.path.exists(p) and os.path.exists(m):
                try:
                    net = cv2.dnn.readNetFromCaffe(p, m)
                    print(f"[INFO] Modelo de edad cargado: proto={os.path.basename(p)} model={os.path.basename(m)}")
                    return net
                except Exception as e:
                    print(f"[WARN] readNetFromCaffe fallo con proto={p} model={m}: {e}")
    return None

age_net = try_load_age_net()
if age_net is None:
    print("[INFO] Modelos de edad no encontrados en models/. (Opcional)")

# Emoción (opcional)
emotion_model = None
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
emotion_path = os.path.join(MODELS_DIR, "emotion_model.hdf5")
if os.path.exists(emotion_path):
    try:
        from tensorflow.keras.models import load_model
        emotion_model = load_model(emotion_path, compile=False)
        print("[INFO] Modelo de emoción cargado (compile=False).")
    except Exception as e:
        print("[WARN] No se pudo cargar modelo de emoción:", e)
else:
    print("[INFO] Modelo de emoción no encontrado en models/. (Opcional)")

# ---------------- Helpers: predicción ----------------
def predict_age(face_bgr):
    if age_net is None:
        return None
    try:
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227,227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        idx = int(preds[0].argmax())
        return AGE_LIST[idx] if idx < len(AGE_LIST) else None
    except Exception as e:
        # si algo falla, deshabilitamos edad para evitar spam de errores
        print("[WARN] predict_age fallo:", e)
        return None

def predict_emotion(face_gray):
    if emotion_model is None:
        return None
    try:
        # ajustar tamaño según input del modelo
        in_shape = emotion_model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        # normalizar y obtener h,w,ch
        dims = tuple(x for x in in_shape if x is not None)
        if len(dims) >= 3:
            h, w, ch = dims[-3], dims[-2], dims[-1]
        elif len(dims) == 2:
            h, w = dims
            ch = 1
        else:
            h, w, ch = 48, 48, 1

        resized = cv2.resize(face_gray, (int(w), int(h)))
        if int(ch) == 1:
            img = resized.astype("float32") / 255.0
            img = np.reshape(img, (1, int(h), int(w), 1))
        else:
            col = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            img = col.astype("float32") / 255.0
            img = np.reshape(img, (1, int(h), int(w), 3))

        preds = emotion_model.predict(img, verbose=0)
        idx = int(np.argmax(preds))
        return emotion_labels[idx] if idx < len(emotion_labels) else None
    except Exception as e:
        print("[WARN] predict_emotion fallo:", e)
        return None

# ---------------- Simple Tracker para caras ----------------
class FaceTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> track dict

    def _centroid(self, bbox):
        x,y,w,h = bbox
        return (int(x + w/2), int(y + h/2))

    def update(self, detections):
        """
        detections: list of bboxes (x,y,w,h)
        Devuelve lista de tuples (id, bbox) correspondientes a detecciones actualizadas
        """
        assigned = {}
        results = []
        centroids = [self._centroid(b) for b in detections]

        # si no hay tracks, crear todas
        if not self.tracks:
            for bbox in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": bbox,
                    "centroid": self._centroid(bbox),
                    "missed": 0,
                    "ages": [],
                    "emotions": []
                }
                results.append((tid, bbox))
            return results

        # construir lista de current track centroids
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]

        # distancia entre detections y tracks
        D = np.zeros((len(track_centroids), len(centroids)), dtype=float) + 1e6
        for i,tc in enumerate(track_centroids):
            for j,dc in enumerate(centroids):
                D[i,j] = np.linalg.norm(np.array(tc) - np.array(dc))

        # greedily match by minimal distance if < threshold
        matched_tracks = set()
        matched_dets = set()
        # iterate in sorted order of distances
        pairs = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                pairs.append((D[i,j], i, j))
        pairs.sort(key=lambda x: x[0])

        for dist,i,j in pairs:
            if i in matched_tracks or j in matched_dets:
                continue
            if dist <= DIST_THRESHOLD:
                tid = track_ids[i]
                bbox = detections[j]
                # update track
                self.tracks[tid]["bbox"] = bbox
                self.tracks[tid]["centroid"] = centroids[j]
                self.tracks[tid]["missed"] = 0
                matched_tracks.add(i)
                matched_dets.add(j)
                results.append((tid, bbox))

        # unmatched detections -> new tracks
        for j, bbox in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": bbox,
                "centroid": centroids[j],
                "missed": 0,
                "ages": [],
                "emotions": []
            }
            results.append((tid, bbox))

        # increment missed for unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["missed"] += 1

        # remove old tracks
        to_delete = [tid for tid, t in self.tracks.items() if t["missed"] > MAX_MISSED_FRAMES]
        for tid in to_delete:
            del self.tracks[tid]

        return results

    def add_prediction(self, tid, age, emotion):
        """Guarda la predicción en el historial del track tid."""
        if tid not in self.tracks:
            return
        if age is not None:
            ages = self.tracks[tid]["ages"]
            ages.append(age)
            if len(ages) > HISTORY_LEN:
                ages.pop(0)
        if emotion is not None:
            emos = self.tracks[tid]["emotions"]
            emos.append(emotion)
            if len(emos) > HISTORY_LEN:
                emos.pop(0)

    def get_stable(self, tid):
        """Devuelve (age_stable, emotion_stable) según moda en historial (o None)."""
        if tid not in self.tracks:
            return (None, None)
        ages = self.tracks[tid]["ages"]
        emos = self.tracks[tid]["emotions"]
        age_stable = None
        emo_stable = None
        if ages:
            age_stable = Counter(ages).most_common(1)[0][0]
        if emos:
            emo_stable = Counter(emos).most_common(1)[0][0]
        return (age_stable, emo_stable)

# ---------------- Main Loop ----------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No puedo abrir la cámara")
        return

    tracker = FaceTracker()
    print("[INFO] Inicio captura. Presioná 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
        # filter small
        detections = [tuple(map(int, d)) for d in detections if d[2]*d[3] >= 1500]

        matched = tracker.update(detections)

        # para cada detección asignada, predecir y guardar
        for tid, bbox in matched:
            x,y,w,h = bbox
            x2, y2 = x+w, y+h
            face_bgr = frame[y:y2, x:x2].copy()
            face_gray = gray[y:y2, x:x2].copy()

            age = predict_age(face_bgr)
            emotion = predict_emotion(face_gray)

            tracker.add_prediction(tid, age, emotion)

            # obtener valores estables
            age_stable, emotion_stable = tracker.get_stable(tid)

            # dibujar
            cv2.rectangle(frame, (x,y), (x2,y2), (0,200,0), 2)
            lines = []
            if age_stable is not None:
                lines.append(f"Edad: {age_stable}")
            if emotion_stable is not None:
                # opcional: mapear emot a 4 categorias (happy, sad, angry, neutral)
                lines.append(f"Emoc: {emotion_stable}")

            for i,ln in enumerate(lines):
                cv2.putText(frame, ln, (x, y - 10 - i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Edad y Emociones (estable) - q para salir", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
