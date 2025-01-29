import numpy as np
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64

# Loading the CSV file and processing data
df = pd.read_csv("/content/Emotion-based-music-recommendation-system/muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):
    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    # Ensure columns are correctly assigned
    if not all(col in data.columns for col in ['name', 'artist', 'link']):
        # Assuming the source DataFrame (df) has these columns
        data['name'] = data['name'] if 'name' in data.columns else df['name']
        data['artist'] = data['artist'] if 'artist' in data.columns else df['artist']
        data['link'] = data['link'] if 'link' in data.columns else df['link']

    # Return the data with the correct columns
    return data[['name', 'artist', 'link']]

def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul

# Load model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('/content/Emotion-based-music-recommendation-system/model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load image
img = cv2.imread('image.jpg')  # Replace with your image path

# Convert the image to grayscale and process it
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

emotion_list = []

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    max_index = int(np.argmax(prediction))
    emotion_list.append(emotion_dict[max_index])

# Process the emotions
processed_emotions = pre(emotion_list)

# Display the detected emotion
if processed_emotions:
    detected_emotion = processed_emotions[0]  # Take the first emotion
    print(f"Emotion detected: {detected_emotion}\n")

    # Get the data for the recommended songs
    recommended_data = fun(processed_emotions)

    # Display the recommendations with Spotify links formatted
    print("Here are some music recommendations based on the detected emotion:\n")
    for link, artist, name in zip(recommended_data["link"], recommended_data['artist'], recommended_data['name']):
        # Check if the link is a Spotify link and convert it if necessary
        if 'spotify' in link:
            track_id = link.split('/')[-1]  # Extract the track ID from the link
            formatted_link = f"https://open.spotify.com/track/{track_id}"
        else:
            formatted_link = link  # If not Spotify, keep the original link

        print(f"Song: {name} - {artist}, Link: {formatted_link}")
else:
    print("No emotion detected. Please try again with a clearer image.")
