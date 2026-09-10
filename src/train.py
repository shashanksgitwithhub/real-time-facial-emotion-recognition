import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.model import build_model


# ==============================
# Configuration
# ==============================

DATA_DIR = "Data/fer2013"
CHECKPOINT_DIR = "checkpoints"

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50


# ==============================
# Main Training Function
# ==============================

def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n========================================")
    print("   FER-2013 Emotion Recognition")
    print("   Improved CNN Training")
    print("========================================\n")

    # --------------------------------
    # Data augmentation
    # --------------------------------

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,

        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,

        shear_range=0.10,
        zoom_range=0.15,

        horizontal_flip=True,

        brightness_range=(0.8, 1.2),

        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # --------------------------------
    # Training data
    # --------------------------------

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),

        target_size=IMAGE_SIZE,
        color_mode="grayscale",

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=True
    )

    # --------------------------------
    # Validation/Test data
    # --------------------------------

    val_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),

        target_size=IMAGE_SIZE,
        color_mode="grayscale",

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=False
    )

    print("\nClass mapping:")
    print(train_gen.class_indices)

    # --------------------------------
    # Calculate class weights
    # --------------------------------

    class_weights_array = compute_class_weight(
        class_weight="balanced",

        classes=np.unique(train_gen.classes),

        y=train_gen.classes
    )

    class_weights = dict(
        enumerate(class_weights_array)
    )

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------
    # Build improved CNN
    # --------------------------------

    print("\nBuilding improved CNN...")

    model = build_model()

    print("\nModel created successfully.\n")

    # --------------------------------
    # Callbacks
    # --------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR,
            "best_model.keras"
        ),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",

        patience=10,

        restore_best_weights=True,

        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-6,

        verbose=1
    )

    # --------------------------------
    # Train
    # --------------------------------

    print("\n========================================")
    print("        STARTING TRAINING")
    print("========================================\n")

    history = model.fit(

        train_gen,

        epochs=EPOCHS,

        validation_data=val_gen,

        class_weight=class_weights,

        callbacks=[
            checkpoint,
            early_stop,
            reduce_lr
        ]
    )

    # --------------------------------
    # Save final model
    # --------------------------------

    final_model_path = os.path.join(
        CHECKPOINT_DIR,
        "final_model.keras"
    )

    model.save(final_model_path)

    # --------------------------------
    # Display results
    # --------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_train_accuracy = max(
        history.history["accuracy"]
    )

    print("\n========================================")
    print("       TRAINING COMPLETED")
    print("========================================")

    print(
        f"\nBest Training Accuracy: "
        f"{best_train_accuracy * 100:.2f}%"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"\nBest model saved to:"
        f"\n{CHECKPOINT_DIR}/best_model.keras"
    )

    print(
        f"\nFinal model saved to:"
        f"\n{CHECKPOINT_DIR}/final_model.keras"
    )

    print("\n========================================\n")


if __name__ == "__main__":
    main()