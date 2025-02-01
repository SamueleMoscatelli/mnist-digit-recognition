# src/evaluate.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

def evaluate_model():
    """
    Load the trained model and evaluate it on the MNIST test dataset.
    """
    # Load the MNIST dataset
    (_, _), (x_test, y_test) = mnist.load_data()

    # Normalize the images (0-255 to 0.0-1.0)
    x_test = x_test / 255.0

    # Reshape images to include channel dimension (28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Load the saved model (now in .keras format)
    model = load_model('models/mnist_model.keras')

    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()