from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)
import mediapipe as mp

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir l'image de BGR à RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processer l'image et détecter les mains
    results = hands.process(rgb_frame)
    
    # Dessiner les annotations des mains
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Afficher l'image annotée
    cv2.imshow('Hand Tracking', frame)
    
    # Sortir de la boucle quand la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def generate_frames():
    while True:
        # Capture une frame de la caméra
        ret, frame = cap.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    # Affiche la page HTML avec la vidéo en direct
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Retourne la réponse avec les frames de la caméra
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
