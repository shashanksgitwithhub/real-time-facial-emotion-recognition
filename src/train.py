import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.model import build_model


# =========================
# Project Configuration
# =========================

DATA_DIR = "Data/fer2013"
CHECKPOINT_DIR = "checkpoints"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# Main Training Function
# =========================

def main():

    print("Starting FER-2013 training...")
    print("Training directory:", TRAIN_DIR)
    print("Testing directory:", TEST_DIR)

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Test data is only normalized
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    # =========================
    # Load Training Dataset
    # =========================

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=True
    )

    # =========================
    # Load Test Dataset
    # =========================

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=False
    )

    print("\nEmotion class mapping:")
    print(train_generator.class_indices)

    # =========================
    # Build CNN Model
    # =========================

    model = build_model()

    model.summary()

    # =========================
    # Callbacks
    # =========================

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR,
            "best_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    # =========================
    # Train Model
    # =========================

    history = model.fit(
        train_generator,
        epochs=40,
        validation_data=test_generator,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    # =========================
    # Save Final Model
    # =========================

    model.save(
        os.path.join(
            CHECKPOINT_DIR,
            "final_model.keras"
        )
    )

    print("\n===================================")
    print("Training completed successfully!")
    print("Best model: checkpoints/best_model.keras")
    print("Final model: checkpoints/final_model.keras")
    print("===================================")


if __name__ == "__main__":
    main()