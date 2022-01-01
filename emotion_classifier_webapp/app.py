import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = './static/uploads'
PREDICTION_FOLDER = './static/predictions'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        make_prediction(image, filename)
        # flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_prediction_image(filename):
    return redirect(url_for('static', filename='predictions/' + filename), code=301)

def face_detector(img):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((48, 48), np.uint8), img
    return (x, w, y, h), roi_gray, img

def make_prediction(frame, filename):
    classifier = load_model('./models/model_v6_23.hdf5')

    emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

    rect, face, image = face_detector(frame)

    face_gray_emo = face.astype("float") / 255.0
    face_gray_emo = img_to_array(face_gray_emo)
    face_gray_emo = np.expand_dims(face_gray_emo, axis=0)

    preds = classifier.predict(face_gray_emo)
    label = str(emotion_classes[preds.argmax()])

    label_position = (rect[0] + int((rect[1] / 2)), abs(rect[2] - 10))
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(app.config['PREDICTION_FOLDER'], filename), image)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
