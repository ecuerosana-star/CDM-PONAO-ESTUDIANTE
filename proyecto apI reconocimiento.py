import os
import cv2 
import time
import argparse
import pickle
import csv
import numpy as np
import face_recognition  # type: ignore
 
# Configuración 
KNOWN_DIR = "known_faces"
ENCODINGS_PKL = "encodings.pkl"
CSV_LOG = "recognition_log.csv"

def build_encodings(known_dir, pkl_path, model="hog"):
    encodings = []
    names = []
    for fname in sorted(os.listdir(known_dir)):
        path = os.path.join(known_dir, fname)
        if not os.path.isfile(path):
            continue
        name, _ = os.path.splitext(fname)
        img = face_recognition.load_image_file(path)
        faces = face_recognition.face_encodings(img)
        if faces:
            encodings.append(faces[0])
            names.append(name)
            print(f"[OK] Loaded: {name}")
        else:
            print(f"[WARN] No face found: {fname}")
    with open(pkl_path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    return encodings, names

def load_encodings(pkl_path):
    if not os.path.exists(pkl_path):
        return [], []
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])

def log_event(csv_path, name, distance):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, name, f"{distance:.4f}"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "name", "distance"])
        writer.writerow(row)

def main(args):
    # Preparar encodings
    encodings, names = load_encodings(args.encodings)
    if not encodings:
        if not os.path.isdir(KNOWN_DIR):
            print("Carpeta known_faces no encontrada. Crea known_faces/ con imágenes de referencia.")
            return
        print("Construyendo encodings desde imágenes en:", KNOWN_DIR)
        encodings, names = build_encodings(KNOWN_DIR, args.encodings, model=args.det_model)
        if not encodings:
            print("No se generaron encodings. Revisa las imágenes en known_faces.")
            return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("No se puede abrir la cámara:", args.camera)
        return

    fps_time = time.time()
    fps = 0
    print("Iniciando. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
        rgb = small[:, :, ::-1]

        t0 = time.time()
        locations = face_recognition.face_locations(rgb, model=args.det_model)
        face_encs = face_recognition.face_encodings(rgb, locations)
        t1 = time.time()

        for (top, right, bottom, left), fe in zip(locations, face_encs):
            # Escalar coordenadas a imagen original
            top = int(top / args.scale)
            right = int(right / args.scale)
            bottom = int(bottom / args.scale)
            left = int(left / args.scale)

            if encodings:
                dists = face_recognition.face_distance(encodings, fe)
                idx = np.argmin(dists)
                best = float(dists[idx])
                name = names[idx] if best < args.threshold else "Desconocido"
            else:
                best = 0.0
                name = "Desconocido"

            # Dibujo discreto y limpio
            color = (50, 200, 50) if name != "Desconocido" else (80, 80, 200)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} {best:.2f}"
            cv2.putText(frame, label, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log y registro
            log_event(CSV_LOG, name, best)
            print(f"[{time.strftime('%H:%M:%S')}] {label} loc=({left},{top},{right},{bottom})")

        # FPS y overlay mínimo
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (time.time() - fps_time)))
        fps_time = time.time()
        overlay = f"{time.strftime('%Y-%m-%d %H:%M:%S')}   FPS:{fps:.1f}"
        cv2.putText(frame, overlay, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Reconocimiento facial - %s" % args.window_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # guardar foto rápida
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print("[SAVE]", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconocimiento facial discreto y profesional")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara")
    parser.add_argument("--scale", type=float, default=0.5, help="Escala para procesamiento (mejor rendimiento)")
    parser.add_argument("--threshold", type=float, default=0.45, help="Umbral de distancia (menor = más estricto)")
    parser.add_argument("--encodings", type=str, default=ENCODINGS_PKL, help="Archivo pickle con encodings")
    parser.add_argument("--det-model", choices=["hog", "cnn"], default="hog", help="Modelo de detección (hog=CPU rápido)")
    parser.add_argument("--window-title", type=str, default="Sistema", help="Título de la ventana")
    args = parser.parse_args()
    main(args)