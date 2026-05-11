import tensorflow as tf

print("Loading model...")

# Load your trained model
model = tf.keras.models.load_model("model.h5")

print("Converting to TFLite format...")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("ai_detector.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model created successfully!")
