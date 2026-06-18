import cv2
import mediapipe as mp
import pygame 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 

pygame.mixer.init()

sounds = [
    pygame.mixer.Sound("#fa.wav"), #indice izquierdo 
    pygame.mixer.Sound("la.wav"), # medio izquierdo
    pygame.mixer.Sound("re.wav"), # anular izquierdo
    pygame.mixer.Sound("#do.wav"), # indice derecho
    pygame.mixer.Sound("#sol.wav"), #medio derecho
    pygame.mixer.Sound("si.wav") #anular derecho
]
def id_finger_down(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=2) as hands:
    finger_state = [False]*6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        #graficar los puntos detectados 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand detection', frame)
        if cv2.waitKey(1) & 0xFF ==27:
            break
    
cap.release()
cv2.destroyAllWindows()