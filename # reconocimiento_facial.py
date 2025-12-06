# reconocimiento_facial.py
import cv2
import os

# Cargar el clasificador preentrenado de rostros de OpenCV
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Crear el reconocedor facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Verificar si existe el modelo entrenado
modelo_path = "modeloLBPH.xml"
if os.path.exists(modelo_path):
    recognizer.read(modelo_path)
    modelo_cargado = True
    print("Modelo de reconocimiento cargado exitosamente")
else:
    modelo_cargado = False
    print("Modelo no encontrado. Solo se realizará detección de rostros.")

# Lista de nombres (debe corresponder con los labels del entrenamiento)
people = ["usuario1"]  # Agrega más nombres según tus datos de entrenamiento

# Abrir la cámara
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convertir a escala de grises (mejora precisión)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Procesar cada rostro detectado
    for (x, y, w, h) in rostros:
        # Extraer la región del rostro
        roi_gris = gris[y:y+h, x:x+w]
        
        if modelo_cargado:
            # Realizar reconocimiento facial
            label, confidence = recognizer.predict(roi_gris)
            
            # Determinar si la persona es conocida o desconocida
            if confidence < 80:  # Umbral de confianza
                # Persona reconocida
                nombre = people[label] if label < len(people) else f"Usuario{label}"
                texto = f"Persona: {nombre}"
                color = (0, 255, 0)  # Verde para persona conocida
                estado = "RECONOCIDA"
            else:
                # Persona no reconocida
                texto = "Persona: Desconocida"
                color = (0, 0, 255)  # Rojo para persona desconocida
                estado = "NO RECONOCIDA"
            
            # Mostrar información completa
            cv2.putText(frame, texto, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Estado: {estado}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Confianza: {int(confidence)}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Solo detección sin reconocimiento
            texto = "Rostro detectado"
            color = (255, 255, 0)  # Amarillo para solo detección
            cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Mostrar información en pantalla
    if modelo_cargado:
        cv2.putText(frame, "Reconocimiento Facial Activo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Solo Deteccion - Sin Modelo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Reconocimiento Facial", frame)

    # Salir con la tecla 'q' o ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cam.release()
cv2.destroyAllWindows()
