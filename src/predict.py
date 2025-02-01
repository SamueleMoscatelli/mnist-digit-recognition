import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def predict_digit(image_path):
    """
    Load the trained model and make predictions on a new image.
    """
    # Load the trained model (now in .keras format)
    model = load_model('models/mnist_model.keras')

    # Load the image
    img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)

    # Normalize the image and reshape it to (1, 28, 28, 1) as required by the model
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    print(f"Predicted digit: {predicted_class}")

    # Optionally, display the image
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Make predictions on new MNIST images.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    # Run the prediction function
    predict_digit(args.image_path)

if __name__ == "__main__":
    main()