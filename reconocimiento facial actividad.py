

     landmarks : list[float] = []                   
          cara = resultado   .multi_face_landmarks[0]  # Primera cara

            # Extraer puntos clave importantes (ojos, nariz, boca, contorno)
            indices_importantes = list(range(0, 468, 7))  # Cada 7 puntos
            # Extraer TODOS los landmarks (468 puntos) para mejor precisión
            for lm in cara.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            for idx in indices_importantes:
                if idx < len(cara.landmark):
                    lm = cara.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
            # Normalizar características
            landmarks = np.array(landmarks)
            landmarks = (landmarks - landmarks.mean()) / (landmarks.std() + 1e-6)

        
         #  clacificar rostros carateristicas. 