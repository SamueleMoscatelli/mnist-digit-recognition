import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import create_model

def train_model():
    """
    Load the MNIST dataset, preprocess the data, and train the CNN model.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images (0-255 to 0.0-1.0)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape images to include channel dimension (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the trained model
    model.save('models/mnist_model.keras')

if __name__ == "__main__":
    train_model()