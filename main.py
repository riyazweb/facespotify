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

# Create an 'uploads' folder for storing uploaded images
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Adjust paths to your CSV and model files:
CSV_PATH = "/content/facespotify/muse_v3.csv"
MODEL_PATH = "/content/facespotify/model.h5"

# Load CSV data for recommendations
df = pd.read_csv(CSV_PATH)
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Divide CSV into segments for each emotion
df_sad    = df[:18000]
df_fear   = df[18000:36000]
df_angry  = df[36000:54000]
df_neutral= df[54000:72000]
df_happy  = df[72000:]

def get_recommendations(emotion_list):
    """
    Given an emotion list (e.g. ["Happy"]), return up to 30 samples
    from the relevant portion of your CSV dataset.
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

    # Ensure columns exist for name, artist, and link
    for col in ["name", "artist", "link"]:
        if col not in data.columns:
            data[col] = ""

    return data[["name", "artist", "link"]]

def process_emotions(emotion_list):
    """
    Collect the unique emotions from detected faces, 
    then return them in a list (e.g. ["Happy", "Neutral"]).
    """
    emotion_counts = Counter(emotion_list)
    return list(emotion_counts.keys())

# Build and load model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the pre-trained weights
model.load_weights(MODEL_PATH)

# Emotion dictionary for mapping model predictions to labels
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
    """
    Render the index.html file from the 'templates' folder.
    Make sure 'index.html' is placed in a 'templates' subfolder.
    """
    return render_template("index.html")

@app.route("/save_image", methods=["POST"])
def save_image():
    """
    This route receives an image (image/jpeg) from the frontend,
    detects faces and emotions, then returns any matched songs.
    """
    # Create a unique subfolder for each new upload
    timestamp = int(time.time())
    upload_dir = os.path.join('uploads', f'image_{timestamp}')
    os.makedirs(upload_dir, exist_ok=True)

    # Store the received image as 'uploaded.jpg'
    image_path = os.path.join(upload_dir, 'uploaded.jpg')
    with open(image_path, "wb") as f:
        f.write(request.data)

    # Read the image via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({
            "message": "Could not read the image.",
            "emotion_found": None,
            "links": []
        }), 400

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_list = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        resized_img = cv2.resize(roi_gray, (48,48))
        expanded_img = np.expand_dims(np.expand_dims(resized_img, axis=-1), axis=0)
        predictions = model.predict(expanded_img, verbose=0)
        max_index = int(np.argmax(predictions))
        emotion_list.append(emotion_dict[max_index])

    # Determine which emotion(s) dominated the detection
    processed = process_emotions(emotion_list)
    if len(processed) == 0:
        return jsonify({
            "message": "No faces or emotions detected.",
            "emotion_found": None,
            "links": []
        })

    # Take just the first recognized emotion (you can adjust as needed)
    detected_emotion = processed[0]

    # Fetch recommended tracks
    rec_data = get_recommendations([detected_emotion])
    results = []
    for link, artist, name in zip(rec_data["link"], rec_data["artist"], rec_data["name"]):
        # Convert if it's a Spotify link
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

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))