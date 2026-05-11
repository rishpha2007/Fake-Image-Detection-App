from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import io
import os

app = Flask(__name__)
CORS(app)

# ================= LOAD MODELS =================
image_model = tf.keras.models.load_model("model.h5")
video_model = tf.keras.models.load_model("video_model.h5")
print("Image Model Input Shape:", image_model.input_shape)
print("Video Model Input Shape:", video_model.input_shape)
# ===== Video Model Dimensions =====
VIDEO_FRAMES = video_model.input_shape[1]
VIDEO_HEIGHT = video_model.input_shape[2]
VIDEO_WIDTH = video_model.input_shape[3]

print("Video expects:", VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH)

# Automatically detect image size from model
if len(image_model.input_shape) == 4:
    IMG_HEIGHT = image_model.input_shape[1]
    IMG_WIDTH = image_model.input_shape[2]
else:
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

# ================= IMAGE FUNCTIONS =================

def preprocess_image(img):
    try:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print("Preprocessing Error:", e)
        return None


def extract_metadata(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        exifdata = image.getexif()

        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            if tag == "Software":
                return exifdata.get(tag_id)

        return "Not Available"
    except:
        return "Not Available"


# ================= VIDEO FUNCTIONS =================

def extract_frames(video_path, max_frames=VIDEO_FRAMES):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return None

    step = max(1, total // max_frames)

    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) > 0:
        return np.array(frames)
    else:
        return None

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template("index.html")


# ---------- IMAGE PREDICTION ----------
# ---------- IMAGE PREDICTION ----------
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files['image']
        file_bytes = file.read()

        img = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"error": "Invalid image file"})

        img_processed = preprocess_image(img)

        if img_processed is None:
            return jsonify({"error": "Image preprocessing failed"})

        prediction = image_model.predict(img_processed)[0][0]
        print("Raw Image Prediction:", prediction)

        # Threshold changed to 0.7
        if prediction > 0.7:
            result = "Fake"
            confidence = round(float(prediction * 100), 2)
        else:
            result = "Real"
            confidence = round(float((1 - prediction) * 100), 2)

        metadata = extract_metadata(file_bytes)

        return jsonify({
            "result": result,
            "confidence": confidence,
            "metadata": metadata
        })

    except Exception as e:
        print("IMAGE ERROR:", e)
        return jsonify({"error": "Image processing failed"})

    except Exception as e:
        print("IMAGE ERROR:", e)
        return jsonify({"error": "Image processing failed"})


# ---------- VIDEO PREDICTION ----------
# ---------- VIDEO PREDICTION ----------
@app.route('/predict_video', methods=['POST'])
def predict_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded"})

        file = request.files['video']
        video_path = "temp_video.mp4"
        file.save(video_path)

        frames = extract_frames(video_path)

        if frames is None:
            os.remove(video_path)
            return jsonify({"error": "Error processing video"})

        frames = np.expand_dims(frames, axis=0)

        prediction = video_model.predict(frames)[0][0]
        print("Raw Video Prediction:", prediction)

        if prediction > 0.5:
            result = "Fake"
            confidence = round(float(prediction * 100), 2)
        else:
            result = "Real"
            confidence = round(float((1 - prediction) * 100), 2)

        os.remove(video_path)


        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        print("VIDEO ERROR:", e)
        return jsonify({"error": str(e)})
# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
