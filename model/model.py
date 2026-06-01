# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    RandomFlip, RandomZoom, RandomRotation,
)

def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        # Define the input shape explicitly. Because the augmentation layers
        # come first, putting input_shape on Conv2D (as before) is ignored by
        # Keras, leaving the model without a defined input shape.
        Input(shape=input_shape),

        # Data augmentation layers (active only during training)
        RandomFlip("horizontal_and_vertical"),
        RandomZoom(-0.3, -0.3),   # negative = zoom in by up to 30%
        RandomRotation(0.2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model