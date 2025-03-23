import os
import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from pyngrok import ngrok

app = Flask(__name__)

# Create uploads folder if not exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# CSV paths (same structure, different content)
CSV_PATH_HINDI = "/content/facespotify/muse_v3.csv"    # Default Hindi file
CSV_PATH_ENGLISH = "/content/facespotify/muse_v3E.csv"  # English file

# MODEL path remains the same
MODEL_PATH = "/content/facespotify/model.h5"

# -------- Load and process Hindi CSV --------
df_hindi = pd.read_csv(CSV_PATH_HINDI)
df_hindi['link'] = df_hindi['lastfm_url']
df_hindi['name'] = df_hindi['track']
df_hindi['emotional'] = df_hindi['number_of_emotion_tags']
df_hindi['pleasant'] = df_hindi['valence_tags']
df_hindi = df_hindi[['name', 'emotional', 'pleasant', 'link', 'artist']]
df_hindi = df_hindi.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
df_hindi_sad     = df_hindi[:18000]
df_hindi_fear    = df_hindi[18000:36000]
df_hindi_angry   = df_hindi[36000:54000]
df_hindi_neutral = df_hindi[54000:72000]
df_hindi_happy   = df_hindi[72000:]

# -------- Load and process English CSV --------
df_english = pd.read_csv(CSV_PATH_ENGLISH)
df_english['link'] = df_english['lastfm_url']
df_english['name'] = df_english['track']
df_english['emotional'] = df_english['number_of_emotion_tags']
df_english['pleasant'] = df_english['valence_tags']
df_english = df_english[['name', 'emotional', 'pleasant', 'link', 'artist']]
df_english = df_english.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
df_english_sad     = df_english[:18000]
df_english_fear    = df_english[18000:36000]
df_english_angry   = df_english[36000:54000]
df_english_neutral = df_english[54000:72000]
df_english_happy   = df_english[72000:]

def get_recommendations(emotion_list, language="hi"):
    """
    Return up to 30 sample tracks based on the detected emotion and language.
    """
    data = pd.DataFrame()
    t = 30  # number of tracks to sample
    emotion = emotion_list[0]
    if language == "en":
        if emotion == "Neutral":
            data = pd.concat([data, df_english_neutral.sample(n=t)], ignore_index=True)
        elif emotion == "Angry":
            data = pd.concat([data, df_english_angry.sample(n=t)], ignore_index=True)
        elif emotion in ["fear", "Fearful"]:
            data = pd.concat([data, df_english_fear.sample(n=t)], ignore_index=True)
        elif emotion in ["happy", "Happy"]:
            data = pd.concat([data, df_english_happy.sample(n=t)], ignore_index=True)
        elif emotion == "Sad":
            data = pd.concat([data, df_english_sad.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_english_angry.sample(n=t)], ignore_index=True)
    else:
        if emotion == "Neutral":
            data = pd.concat([data, df_hindi_neutral.sample(n=t)], ignore_index=True)
        elif emotion == "Angry":
            data = pd.concat([data, df_hindi_angry.sample(n=t)], ignore_index=True)
        elif emotion in ["fear", "Fearful"]:
            data = pd.concat([data, df_hindi_fear.sample(n=t)], ignore_index=True)
        elif emotion in ["happy", "Happy"]:
            data = pd.concat([data, df_hindi_happy.sample(n=t)], ignore_index=True)
        elif emotion == "Sad":
            data = pd.concat([data, df_hindi_sad.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_hindi_angry.sample(n=t)], ignore_index=True)

    # Ensure necessary columns exist
    for col in ["name", "artist", "link"]:
        if col not in data.columns:
            data[col] = ""
    return data[["name", "artist", "link"]]

def process_emotions(emotion_list):
    """
    Return unique emotions from detected faces.
    """
    from collections import Counter
    emotion_counts = Counter(emotion_list)
    return list(emotion_counts.keys())

# -------- Build and load model --------
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
model.load_weights(MODEL_PATH)

# Emotion dictionary for mapping predictions to labels
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
    """
    Receive image, detect emotion, and return recommendations.
    """
    timestamp = int(time.time())
    upload_dir = os.path.join('uploads', f'image_{timestamp}')
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, 'uploaded.jpg')
    with open(image_path, "wb") as f:
        f.write(request.data)

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({
            "message": "Could not read the image.",
            "emotion_found": None,
            "links": []
        }), 400

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
    language = request.headers.get("X-Language", "hi")
    rec_data = get_recommendations([detected_emotion], language)
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

@app.route("/change_language", methods=["POST"])
def change_language():
    """
    Return updated recommendations using the stored emotion and new language.
    """
    data = request.get_json()
    detected_emotion = data.get("emotion")
    language = data.get("language", "hi")
    rec_data = get_recommendations([detected_emotion], language)
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
        "links": results
    })

if __name__ == "__main__":
    ngrok.set_auth_token("2UUGMJW8gaZ7Ikrl53By3xYHdLs_6b3ipRxC3rEXwy7JgQv5Y")
    public_url = ngrok.connect(5000, domain="prepared-singularly-shepherd.ngrok-free.app")
    print(f"🌐 Ngrok URL: {public_url}")
    app.run(debug=False, port=5000)
