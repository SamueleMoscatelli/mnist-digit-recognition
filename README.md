# MNIST Digit Recognition with TensorFlow

This project implements a **handwritten digit recognition** model using the **MNIST dataset**. The model is built with **TensorFlow** and **Keras** (high-level API for TensorFlow), and uses **Convolutional Neural Networks (CNNs)** for classifying images of handwritten digits (0-9).

## Table of Contents

- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)

## Project Description

The goal of this project is to build a CNN-based classifier capable of recognizing handwritten digits from the MNIST dataset. The project will cover data preprocessing, model building, training, and evaluation. The model will be trained using **TensorFlow** and **Keras**, and the trained model will be saved in the `.keras` format for easy deployment or further fine-tuning.

The project includes:
✅ **Data preprocessing**  
✅ **Model training and evaluation**  
✅ **Flexible image prediction** – Accepts images of **any size** and resizes them correctly  
✅ **Model saving in `.keras` format** for easy deployment 

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/SamueleMoscatelli/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Training the Model
To train the model, simply run the following command:
```bash
python train_model.py
```

This will load the MNIST dataset, preprocess the data, build the CNN model, and start training. The model will be saved as model `.keras` upon completion.

## Model Architecture
- Input Layer: 28x28 pixel grayscale images from the MNIST dataset.
- Conv2D Layers: For extracting features from images.
- MaxPooling Layers: To reduce dimensions and prevent overfitting.
- Dense Layer: Fully connected layer for classification.
- Softmax Output: 10 output classes for digits 0-9.

## Model Evaluation
Once the model is trained, it can be evaluated using:
```bash
python evaluate_model.py
```
This will load the trained model and evaluate its performance on the test dataset. The accuracy will be printed to the terminal.

## Usage
To make predictions on new images, use the following script:
```bash
python predict_digit.py --image_path path_to_image
```

The script automatically resizes images to 28x28 while keeping the aspect ratio.
It converts to grayscale if needed.
The image does NOT need to be preprocessed manually!
✅ Supports any input image dimensions
✅ Handles color images by converting to grayscale

Example
If you have a digit image named "my_digit.png", run:

```bash
python src/predict.py --image_path images/my_digit.png
```