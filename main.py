import os
import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

app = Flask(__name__)

# 📁 Create uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# 🔍 Set paths for CSV and model files (make sure these files are in your project)
CSV_PATH = "muse_v3.csv"
MODEL_PATH = "model.h5"

# 🎵 Load CSV data for recommendations
try:
    df = pd.read_csv(CSV_PATH)
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']
    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    # Divide CSV into segments for each emotion
    df_sad     = df[:18000]
    df_fear    = df[18000:36000]
    df_angry   = df[36000:54000]
    df_neutral = df[54000:72000]
    df_happy   = df[72000:]
except Exception as e:
    print("Error loading CSV:", e)
    df = None

def get_recommendations(emotion_list):
    """
    Return up to 30 song recommendations based on the emotion.
    """
    data = pd.DataFrame()
    if len(emotion_list) == 1:
        emotion = emotion_list[0]
        t = 30  # number of tracks to sample
        if emotion == "Neutral":
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif emotion == "Angry":
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif emotion in ["fear", "Fearful"]:
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif emotion in ["happy", "Happy"]:
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        elif emotion == "Sad":
            data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    # Ensure columns exist
    for col in ["name", "artist", "link"]:
        if col not in data.columns:
            data[col] = ""
    return data[["name", "artist", "link"]]

def process_emotions(emotion_list):
    """Return a list of unique emotions."""
    return list(Counter(emotion_list).keys())

# 🤖 Build and load the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

try:
    model.load_weights(MODEL_PATH)
except Exception as e:
    print("Error loading model weights:", e)

# Mapping model output to emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save_image", methods=["POST"])
def save_image():
    try:
        # ✅ Check if image data is received
        if not request.data:
            return jsonify({"error": "No image data received"}), 400

        # 📅 Create unique folder and save image
        timestamp = int(time.time())
        upload_dir = os.path.join('uploads', f'image_{timestamp}')
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, 'uploaded.jpg')
        with open(image_path, "wb") as f:
            f.write(request.data)

        # 📸 Read the image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"error": "Could not read the image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_list = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(roi_gray, (48, 48))
            expanded_img = np.expand_dims(np.expand_dims(resized_img, axis=-1), axis=0)
            predictions = model.predict(expanded_img, verbose=0)
            max_index = int(np.argmax(predictions))
            emotion_list.append(emotion_dict[max_index])

        processed = process_emotions(emotion_list)
        if len(processed) == 0:
            return jsonify({
                "message": "No faces or emotions detected.",
                "emotion_found": None,
                "links": []
            })

        detected_emotion = processed[0]
        rec_data = get_recommendations([detected_emotion])
        results = []
        for link, artist, name in zip(rec_data["link"], rec_data["artist"], rec_data["name"]):
            if "spotify" in link:
                track_id = link.split("/")[-1]
                link = f"https://open.spotify.com/track/{track_id}"
            results.append({
                "song": name,
                "artist": artist,
                "link": link
            })

        return jsonify({
            "message": f"Received image. Detected emotion: {detected_emotion}",
            "emotion_found": detected_emotion,
            "links": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
