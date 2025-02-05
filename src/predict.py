import tensorflow as tf
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(28, 28)):
    """
    Load an image, convert to grayscale, resize while keeping aspect ratio, and normalize.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if img is None:
        raise ValueError("Error loading image. Check the file path.")

    # Get original dimensions
    original_size = img.shape[:2]  # (height, width)

    # Resize while keeping aspect ratio
    img = resize_with_aspect_ratio(img, target_size)

    # Normalize pixel values (0-1 range)
    img = img / 255.0  

    # Reshape to fit the model input shape (1, 28, 28, 1)
    img = np.expand_dims(img, axis=(0, -1))  

    return img, original_size

def resize_with_aspect_ratio(img, target_size):
    """
    Resize an image while keeping its aspect ratio and padding if needed.
    """
    h, w = img.shape  # Original height and width
    target_h, target_w = target_size

    # Compute the scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas of the target size (28x28)
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255  # White background

    # Compute centering position
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place resized image on the blank canvas
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def predict_digit(image_path):
    """
    Load the trained model and make predictions on a new image.
    """
    # Load the trained model
    model = load_model('models/mnist_model.keras')

    # Preprocess the image
    img, original_size = preprocess_image(image_path)

    # Make the prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    print(f"Predicted digit: {predicted_class}")

    # Display the image with prediction
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Make predictions on new images.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    # Run the prediction function
    predict_digit(args.image_path)

if __name__ == "__main__":
    main()
