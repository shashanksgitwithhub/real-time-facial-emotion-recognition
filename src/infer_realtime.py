import cv2
import numpy as np
import tensorflow as tf


# =========================
# Emotion Labels
# =========================

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# =========================
# Load Trained Model
# =========================

MODEL_PATH = "checkpoints/best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# Face Detection
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================
# Start Webcam
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


# =========================
# Real-Time Detection Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert webcam frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Extract face region
        roi = gray[y:y + h, x:x + w]

        # Resize to model input size
        roi = cv2.resize(
            roi,
            (48, 48)
        )

        # Normalize pixel values
        roi = roi.astype("float32") / 255.0

        # Reshape for CNN
        roi = np.reshape(
            roi,
            (1, 48, 48, 1)
        )

        # Predict emotion
        predictions = model.predict(
            roi,
            verbose=0
        )[0]

        # Get highest probability
        emotion_index = np.argmax(predictions)

        emotion = EMOTION_LABELS[emotion_index]

        confidence = predictions[emotion_index] * 100

        # Display label
        label = f"{emotion}: {confidence:.1f}%"

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display webcam window
    cv2.imshow(
        "Real-Time Emotion Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()

print("Emotion recognition stopped.")