#!/usr/bin/python
"""
Assignment 3  - Document classification using pretrained image embeddings
"""

# Importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

######
# Defining functions
######

# Plotting
def plot_history(H, epochs, output_dir):
    """
    Function to plot loss and accuracy curves.

    Parameters:
        H (History): The history object returned from model.fit.
        epochs (int): Number of epochs.
        output_dir (str): Output directory to save the plots.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    # Save the plots
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.show()

# Loading data
def load_data(data_path):
    """
    Load image data from the specified path.

    Args:
    - data_path (str): Path to image data.

    Returns:
    - images (list): List of dictionaries containing image data and labels.
    """
    dirs = sorted(os.listdir(data_path))
    images = []

    for directory in dirs:
        subfolder = os.path.join(data_path, directory)
        filenames = sorted(os.listdir(subfolder))

        for image in filenames:
            file_path = os.path.join(subfolder, image)
            if file_path.endswith('.jpg'):
                image_data = load_img(file_path, target_size=(224, 224))
                images.append({"label": directory, "image": image_data})

    return images

# Preprocess images
def preprocess_images(images):
    """
    Preprocess images for VGG model.

    Args:
    - images (list): List of image data.

    Returns:
    - processed_images (list): List of preprocessed images.
    """
    image_arrays = [img_to_array(image['image']) for image in images]
    image_reshaped = [image_array.reshape((image_array.shape[0], image_array.shape[1], image_array.shape[2])) for image_array in image_arrays]
    processed_images = [preprocess_input(image_reshape) for image_reshape in image_reshaped]

    return processed_images

# Building model
def build_model(num_classes):
    """
    Build and compile the VGG model with new classification layers.

    Args:
    - num_classes (int): Number of classes.

    Returns:
    - model (Model): Compiled Keras model.
    """
    base_model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    flat1 = Flatten()(base_model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)

    model = Model(inputs=base_model.inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

###### 
# Main function
######

def main(data_path, output_dir):
    """
    Main function to load data, build and train the model, and save results.

    Args:
    - data_path (str): Path to image data.
    - output_dir (str): Directory to save results.

    Returns:
    - None
    """
    # Load data
    images = load_data(data_path)
    processed_images = preprocess_images(images)

    # Labels
    labelNames = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform([image["label"] for image in images])

    # Split data into train and test sets with stratification by label
    X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Build and compile the model
    model = build_model(len(labelNames))

    # Train the model
    H = model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, batch_size=128, epochs=10)

    # Plot training history
    plot_history(H, 10, output_dir)

    # Evaluate the model
    predictions = model.predict(np.array(X_test), batch_size=128)
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=labelNames)
    print(report)

    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document classification using pretrained image embeddings")
    parser.add_argument("--data_path", type=str, default="in/Tobacco3482", help="Path to image data")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to save results")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)