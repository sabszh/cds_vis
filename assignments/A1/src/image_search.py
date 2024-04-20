#!/usr/bin/python

"""
Assignment 1 - Building a simple image search algorithm
"""
# Importing packages
import os
import sys
import cv2
import numpy as np
from numpy.linalg import norm
import pandas as pd
import argparse
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.join(".."))

# Defining function for extracting the color histogram for an image 
def extract_color_hist(image_path):
    """
    Extracts color histograms for a single image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: A numpy array containing histograms for each channel (red, green, blue).
    """
    
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Split channels
    channels = cv2.split(image)

    histograms = [cv2.calcHist([channel], [0], None, [255], [0, 256]) for channel in channels]
    histograms = np.concatenate(histograms)

    # Normalize histogram
    histograms = cv2.normalize(histograms, histograms, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return histograms

# Making function for comparing histograms
def compare_histograms(target_histogram, histograms_list):
    """
    Compares the histogram of a target image to other histograms using Chi-Squared distance.

    Args:
        target_histogram (numpy.ndarray): Histogram of the target image.
        histograms_list (list): List containing tuples of (filename, histograms) for all images.

    Returns:
        list: A list of tuples where each tuple contains the filename and the Chi-Squared distance
              between the target image histogram and the histogram of each image in the dataset.
    """
    distances = [(filename, round(cv2.compareHist(target_histogram, histogram, cv2.HISTCMP_CHISQR), 2))
                 for filename, histogram in histograms_list]

    return distances

# Using VGG16 and KNN
def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=False)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    return normalized_features

def find_similar_images(target_image_path, dataset_dir, num_neighbors=5):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    target_features = extract_features(target_image_path, model)

    filenames = [os.path.join(dataset_dir, name) for name in sorted(os.listdir(dataset_dir))]

    feature_list = []
    for filename in tqdm(filenames, desc="Extracting features"):
        feature_list.append(extract_features(filename, model))

    neighbors = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='brute', metric='cosine').fit(feature_list)

    distances, indices = neighbors.kneighbors([target_features])

    similar_images = []
    for i in range(1, num_neighbors + 1):
        similar_images.append(filenames[indices[0][i]])

    return similar_images

# Function to save results to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns = ["Filename", "Distance"])
    df.to_csv(os.path.join(OUTPUT_FOLDER, filename), index = False)

# Main function

def main():
    # Defining folder paths
    INPUT_FOLDER = os.path.join("in", "flowers")
    OUTPUT_FOLDER = os.path.join("..","out")

    # Extract histograms for all images in the folder
    histograms_list = [(image_filename, extract_color_hist(os.path.join(INPUT_FOLDER, image_filename)))
                       for image_filename in os.listdir(INPUT_FOLDER)]

    # Define path for target image
    target_image_path = os.path.join(INPUT_FOLDER, "image_1009.jpg")

    # Extract histogram for the target image
    target_histogram = extract_color_hist(target_image_path)

    # Compare histograms
    distances = compare_histograms(target_histogram, histograms_list)

    # Sort the distances list based on the distances (second element of each tuple)
    top_5_closest = sorted(distances, key=lambda x: x[1])[:5]

    # Save results to CSV
    save_to_csv(top_5_closest, "5TOP_similar_images.csv")

    print("CSV file saved in the out folder.")

if __name__ == "__main__":
    main()