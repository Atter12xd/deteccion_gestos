import cv2
import mediapipe as mp

# Cargar el modelo de reconocimiento de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video (cámara)
cap = cv2.VideoCapture(0)

# Iniciar detector de manos
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo acceder a la cámara.")
            break

        # Convertir imagen a RGB para procesamiento con MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Dibujar los landmarks de la mano si hay manos detectadas
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer landmarks de la primera mano detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]

            # Identificación de gestos simples basados en los landmarks

            # 1. Gesto: Pulgar hacia arriba
            thumb_tip = landmarks_list[4]  # Punta del pulgar
            index_tip = landmarks_list[8]  # Punta del dedo índice

            # 2. Gesto: Paz (✌️)
            index_tip = landmarks_list[8]
            middle_tip = landmarks_list[12]

            # 3. Gesto: Dedo índice (🖐️)
            thumb_tip = landmarks_list[4]
            middle_tip = landmarks_list[12]
            ring_tip = landmarks_list[16]
            pinky_tip = landmarks_list[20]

            # 4. Gesto: Puño cerrado
            # Verifica si los dedos están doblados (las puntas están cerca de la muñeca)
            if (landmarks_list[4][1] > landmarks_list[3][1] and  # Pulgar
                landmarks_list[8][1] > landmarks_list[7][1] and  # Índice
                landmarks_list[12][1] > landmarks_list[11][1] and  # Medio
                landmarks_list[16][1] > landmarks_list[15][1] and  # Anular
                landmarks_list[20][1] > landmarks_list[19][1]):  # Meñique
                cv2.putText(image, 'Gesto: quieres pelea?', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            else:
                # 5. Gesto de Paz
                if index_tip[1] < middle_tip[1] and index_tip[0] < middle_tip[0]:
                    cv2.putText(image, 'Gesto: Paz', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 6. Gesto de Pulgar Arriba
                elif thumb_tip[1] < index_tip[1]:  # Pulgar por encima del índice
                    cv2.putText(image, 'Gesto: Ok', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 7. Gesto de Dedo Índice
                elif index_tip[1] < thumb_tip[1]:  # El índice está extendido hacia arriba
                    cv2.putText(image, 'Gesto: Para', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 8. Gesto: Dedos Estirados (todos los dedos estirados)
                elif (landmarks_list[4][1] < landmarks_list[3][1] and  # Pulgar
                    landmarks_list[8][1] < landmarks_list[7][1] and  # Índice
                    landmarks_list[12][1] < landmarks_list[11][1] and  # Medio
                    landmarks_list[16][1] < landmarks_list[15][1] and  # Anular
                    landmarks_list[20][1] < landmarks_list[19][1]):  # Meñique
                    cv2.putText(image, 'Gesto: Dedos Estirados', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen con la cámara en tiempo real
        cv2.imshow('Gesto Reconocimiento', image)

        # Detener si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cerrar la cámara
cap.release()
cv2.destroyAllWindows()
