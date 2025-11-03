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

# --- CONFIGURATION ---
CSV_PATH = "/content/facespotify/spotify_tracks.csv" 
MODEL_PATH = "/content/facespotify/model.h5"

LANGUAGE_MAP = {
    "hindi": "hindi",
    "english": "english",
    "tamil": "tamil",
    "telugu": "telugu"
}

# --- DATA LOADING AND PREPROCESSING ---
print("Loading and processing dataset...")
try:
    df = pd.read_csv(CSV_PATH)
    df['name'] = df['track_name']
    df['artist'] = df['artist_name']
    df['link'] = df['track_url']
    df['language'] = df['language'].str.lower().str.strip()
    df = df.sort_values(by="valence").reset_index(drop=True)
    df = df[['name', 'artist', 'link', 'language', 'valence']]
    print("\n" + "---" * 10)
    print("VERIFICATION STEP: Found these language codes in CSV:", df['language'].unique())
    print("---" * 10 + "\n")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: The CSV file was not found at the path: {CSV_PATH}")
    exit()
except KeyError as e:
    print(f"FATAL ERROR: A required column is missing from the CSV: {e}")
    exit()


# --- Get Recommendations Function (with debugging) ---
def get_recommendations(emotion_list, language_code="hi"):
    # (This function is kept as is from the previous version)
    print("\n" + "---" * 10)
    print("DEBUGGING INSIDE get_recommendations")
    print(f"--> Received Emotion: {emotion_list[0] if emotion_list else 'None'}")
    print(f"--> Received Language Code: '{language_code}'")
    if not emotion_list: return pd.DataFrame(columns=["name", "artist", "link"])
    emotion = emotion_list[0]
    lang_df = df[df['language'] == language_code]
    print(f"--> STEP 1: Found {len(lang_df)} songs for language '{language_code}'.")
    if lang_df.empty: return pd.DataFrame(columns=["name", "artist", "link"])
    n_tracks, band_size = len(lang_df), len(lang_df) // 5
    print(f"--> STEP 2: Slicing for emotion '{emotion}'. Num songs: {n_tracks}, Band size: {band_size}")
    emotion_df = pd.DataFrame()
    if emotion in ["Sad", "Angry"]: emotion_df = lang_df.iloc[:band_size]
    elif emotion in ["fear", "Fearful", "Disgusted"]: emotion_df = lang_df.iloc[band_size:2*band_size]
    elif emotion == "Neutral": emotion_df = lang_df.iloc[2*band_size:3*band_size]
    elif emotion in ["happy", "Happy", "Surprised"]: emotion_df = lang_df.iloc[-band_size:]
    else: emotion_df = lang_df.iloc[2*band_size:3*band_size]
    print(f"   - Found {len(emotion_df)} songs in the emotional slice.")
    num_to_sample = min(30, len(emotion_df))
    if num_to_sample == 0: return pd.DataFrame(columns=["name", "artist", "link"])
    recommended_songs = emotion_df.sample(n=num_to_sample)
    print(f"--> STEP 3: Successfully sampled {len(recommended_songs)} songs.")
    print("---" * 10 + "\n")
    return recommended_songs

def process_emotions(emotion_list):
    return list(Counter(emotion_list).keys())

# --- Build and load model ---
print("Loading emotion detection model...")
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
print("Model loaded successfully.")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save_image", methods=["POST"])
def save_image():
    """Receive image, detect emotion, and return recommendations."""
    # --- FIX 1: RESTORED FULL IMAGE HANDLING LOGIC ---
    timestamp = int(time.time())
    upload_dir = os.path.join('uploads', f'image_{timestamp}')
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, 'uploaded.jpg')
    with open(image_path, "wb") as f:
        f.write(request.data)

    img = cv2.imread(image_path)
    
    # --- FIX 2: ADDED CHECK FOR INVALID IMAGE ---
    # This prevents a crash if the received data isn't a proper image.
    if img is None:
        return jsonify({
            "message": "Could not read the uploaded image. It might be corrupted.",
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
    
    # --- FIX 3: ADDED CHECK FOR NO DETECTED FACES ---
    # This is the main fix for the 500 error. It handles the case where the `processed` list is empty.
    if not processed:
        return jsonify({
            "message": "No faces or emotions detected. Please try again.",
            "emotion_found": None,
            "links": []
        })

    detected_emotion = processed[0] # This line is now safe
    language_name = request.headers.get("X-Language", "hindi").lower()
    language = LANGUAGE_MAP.get(language_name, "hindi")
    
    rec_data = get_recommendations([detected_emotion], language)
    results = []
    
    for _, row in rec_data.iterrows():
        results.append({"song": row['name'], "artist": row['artist'], "link": row['link']})

    return jsonify({
        "message": f"Received image. Detected emotion: {detected_emotion}",
        "emotion_found": detected_emotion,
        "links": results
    })


@app.route("/change_language", methods=["POST"])
def change_language():
    """Return updated recommendations using the stored emotion and new language."""
    data = request.get_json()
    detected_emotion = data.get("emotion")
    language_name = data.get("language", "hindi").lower()
    language = LANGUAGE_MAP.get(language_name, "hindi")
    
    if not detected_emotion:
        return jsonify({"links": [], "message": "Emotion not provided."})
        
    rec_data = get_recommendations([detected_emotion], language)
    results = []
    
    for link, artist, name in zip(rec_data["link"], rec_data["artist"], rec_data["name"]):
        results.append({"song": name, "artist": artist, "link": link})
        
    return jsonify({"links": results})

 
if __name__ == "__main__":
    ngrok.set_auth_token("2UUGMJW8gaZ7Ikrl53By3xYHdLs_6b3ipRxC3rEXwy7JgQv5Y")
    # Make sure to use a static domain if you have a paid ngrok plan for consistency
    public_url = ngrok.connect(5000, domain="prepared-singularly-shepherd.ngrok-free.app")
    print(f"🌐 Ngrok URL: {public_url}")
    # Set debug=False for production or when sharing the link
    app.run(debug=False, port=5000)
