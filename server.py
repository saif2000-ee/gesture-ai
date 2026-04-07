from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from collections import deque
import os

app = Flask(__name__)

# =========================
# تحميل الموديلات
# =========================
model_letters_dict = pickle.load(open('model_letters.p', 'rb'))
model_letters = model_letters_dict['model']
max_length_letters = model_letters_dict.get('max_length', 42)

model_words_dict = pickle.load(open('model_words.p', 'rb'))
model_words = model_words_dict['model']
max_length_words = model_words_dict.get('max_length', 42)

# =========================
# MediaPipe Hands (أخف وأسرع)
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0   # 🔥 مهم للتسريع
)

# =========================
# تثبيت النتيجة
# =========================
history = deque(maxlen=5)

# =========================
# تقليل الضغط (frame skipping)
# =========================
frame_counter = 0

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global frame_counter

    try:
        frame_counter += 1

        # 🔥 تجاهل بعض الفريمات لتخفيف الضغط
        if frame_counter % 2 != 0:
            return jsonify({'prediction': ''})

        data = request.json['image']
        mode = request.json.get('mode', 'words')

        # اختيار الموديل
        if mode == 'letters':
            model = model_letters
            max_length = max_length_letters
        else:
            model = model_words
            max_length = max_length_words

        # =========================
        # تحويل الصورة
        # =========================
        img_data = base64.b64decode(data.split(',')[1])
        npimg = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'prediction': '?'})

        # 🔥 تصغير الصورة (أهم تحسين)
        frame = cv2.resize(frame, (320, 240))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # =========================
        # استخراج النقاط
        # =========================
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_), min(y_)

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            # padding
            if len(data_aux) < max_length:
                data_aux.extend([0] * (max_length - len(data_aux)))

            input_data = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)

            # =========================
            # prediction
            # =========================
            prediction = model.predict(input_data)
            predicted_character = str(prediction[0])

            # =========================
            # smoothing
            # =========================
            history.append(predicted_character)
            final_prediction = max(set(history), key=history.count)

            return jsonify({'prediction': final_prediction})

        else:
            return jsonify({'prediction': '?'})

    except Exception as e:
        print("Error:", e)
        return jsonify({'prediction': '?'})


# =========================
# تشغيل السيرفر (مهم للـ Deploy)
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
