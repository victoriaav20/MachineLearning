import os
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Charger le modèle de reconnaissance des signes
model = load_model('model_sign_language.h5')

# Variable globale pour la prédiction courante
current_prediction = ""

def prediction(filename):
    global current_prediction
    
    # Charger et prétraiter l'image à comparer
    img_path = filename
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normaliser les pixels entre 0 et 1

    # Faire la prédiction
    prediction = model.predict(img_array)

    # Afficher la prédiction
    predicted_class_index = np.argmax(prediction)
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    predicted_class = classes[predicted_class_index]
    current_prediction = predicted_class  # Mettre à jour la variable globale
    print("La classe prédite de l'img", filename, " est :", predicted_class)

def extract_hand_roi(image, landmarks):
    h, w, _ = image.shape
    landmark_coords = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks.landmark]
    
    x_coords, y_coords = zip(*landmark_coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    margin = 60
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)
    
    hand_roi = image[y_min:y_max, x_min:x_max]
    hand_roi_resized = cv2.resize(hand_roi, (200, 200))
    return hand_roi_resized


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_roi = extract_hand_roi(frame, hand_landmarks)
                    
                    timestamp = int(time.time())
                    filename = f"captured_frame_{timestamp}.jpg"
                    
                    cv2.imwrite(filename, hand_roi)
                    prediction(filename)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global current_prediction
    return jsonify(prediction=current_prediction)

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
