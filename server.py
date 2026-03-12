from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# ===== تحميل الموديل مرة واحدة عند start السيرفر =====
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Mediapipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ===== الصفحة الرئيسية =====
@app.route('/')
def index():
    return render_template('index.html')

# ===== استلام صورة من الويب وتحليلها =====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image'].split(',')[1]  # إزالة prefix
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Mediapipe hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    letter = ""
    if results.multi_hand_landmarks:
        # تحويل landmarks لمصفوفة للتنبؤ
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        X = np.array(landmarks).reshape(1, -1)
        letter = model.predict(X)[0]

    return jsonify({'letter': letter})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
