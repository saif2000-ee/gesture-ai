from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

# تحميل الموديل
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image'].split(',')[1]  # base64 بعد 'data:image/jpeg;base64,'
    img_bytes = base64.b64decode(img_data)

    # تحويل الصورة إلى numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Mediapipe
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            X = np.array(landmarks).reshape(1, -1)
            letter = model.predict(X)[0]
            return jsonify({'letter': letter})
        else:
            return jsonify({'letter': ''})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
