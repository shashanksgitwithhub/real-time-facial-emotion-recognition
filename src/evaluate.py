import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


DATA_DIR = "Data/fer2013"
MODEL_PATH = "checkpoints/best_model.keras"

TEST_DIR = os.path.join(DATA_DIR, "test")


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def main():

    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=False
    )

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        test_generator,
        verbose=1
    )

    print("\n==============================")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("==============================")

    # Predictions
    predictions = model.predict(
        test_generator,
        verbose=1
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_classes = test_generator.classes

    # Classification report
    print("\nClassification Report:\n")

    print(
        classification_report(
            true_classes,
            predicted_classes,
            target_names=EMOTION_LABELS
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        true_classes,
        predicted_classes
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Display confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EMOTION_LABELS
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title("FER-2013 Emotion Recognition Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "checkpoints/confusion_matrix.png"
    )

    plt.show()

    print("\nConfusion matrix saved to:")
    print("checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    main()