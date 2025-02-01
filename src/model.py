import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """
    Creates and returns a CNN model for handwritten digit recognition (MNIST).
    """
    model = models.Sequential()

    # Add Conv2D layers with MaxPooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the layers before feeding into dense layer
    model.add(layers.Flatten())

    # Add Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # Output layer (10 classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model