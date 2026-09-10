import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2


def build_model(input_shape=(48, 48, 1), num_classes=7):

    model = Sequential([

        # Block 1
        Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001),
            input_shape=input_shape
        ),
        BatchNormalization(),

        Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),

        Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),

        # Block 3
        Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),

        Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),

        # Classification
        Flatten(),

        Dense(
            512,
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        BatchNormalization(),
        Dropout(0.5),

        # Seven emotions
        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model