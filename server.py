from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

# تحميل المودال
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
max_length = model_dict.get('max_length', 42)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2)

labels_dict = {0:'ع',1:'ال',2:'ا',3:'ب',4:'ض',5:'د',6:'ف',7:'غ',8:'ح',9:'ه',10:'ج',11:'ك',12:'خ',13:'لا',14:'ل',15:'م',16:'ن',17:'ق',18:'ر',19:'ص',20:'س',21:'ش',22:'ط',23:'ت',24:'ة',25:'ذ',26:'ث',27:'و',28:'ي',29:'ظ',30:'ز',}

# 🔴 هذا هو الحل
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    data = request.json['image']

    img_data = base64.b64decode(data.split(',')[1])
    npimg = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    H,W,_ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux=[]
    x_=[]
    y_=[]

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x-min(x_))
                data_aux.append(lm.y-min(y_))

        if len(data_aux)<max_length:
            data_aux.extend([0]*(max_length-len(data_aux)))

        input_data=np.asarray(data_aux).reshape(1,-1)

        prediction=model.predict(input_data)

        predicted_character=labels_dict[int(prediction[0])]

        return jsonify({'prediction':predicted_character})

    return jsonify({'prediction':'?'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
