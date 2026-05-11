import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
IMG_SIZE = 128
SEQUENCE_LENGTH = 10
DATASET_PATH = "datasets/video/train"

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < SEQUENCE_LENGTH:
        cap.release()
        return None

    for i in range(SEQUENCE_LENGTH):
        frame_number = int(frame_count / SEQUENCE_LENGTH * i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    if len(frames) == SEQUENCE_LENGTH:
        return np.array(frames)
    return None

# ---------------- LOAD DATA ----------------
def load_data():
    X = []
    y = []

    for label in ['real', 'fake']:
        folder = os.path.join(DATASET_PATH, label)
        for file in os.listdir(folder):
            video_path = os.path.join(folder, file)

            frames = extract_frames(video_path)

            if frames is not None:
                X.append(frames)
                y.append(0 if label == 'real' else 1)

    return np.array(X), np.array(y)

print("Loading video dataset...")
X, y = load_data()

print("Dataset shape:", X.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = models.Sequential()

# CNN feature extractor applied to each frame
model.add(layers.TimeDistributed(
    layers.Conv2D(32, (3,3), activation='relu'),
    input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
model.add(layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
model.add(layers.TimeDistributed(layers.Flatten()))

# LSTM
model.add(layers.LSTM(64))

# Output
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
print("Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=2
)

# ---------------- SAVE ----------------
model.save("video_model.h5")
print("✅ Video model saved as video_model.h5")
