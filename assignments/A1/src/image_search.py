#!/usr/bin/python
"""
Assignment: 1 - Building a simple image search algorithm
Course: Visual Analytics
Author: Sabrina Zaki Hansen
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

########
## Defining needed functions
########

# Argument parsing
def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Image search algorithm")
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("--method", choices=["histogram", "vgg"], default="histogram", help="Method for image search")
    args = parser.parse_args()

    return args

# Defining histogram class
class HistogramImageSearch:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.histograms_list = self.extract_histograms()

    def extract_histogram(self, image_path):
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

    def extract_histograms(self):
        histograms_list = [(image_filename, self.extract_histogram(os.path.join(self.dataset_dir, image_filename)))
                           for image_filename in os.listdir(self.dataset_dir)]
        return histograms_list

    def compare_histograms(self, target_histogram):
        distances = [(filename, round(cv2.compareHist(target_histogram, histogram, cv2.HISTCMP_CHISQR), 2))
                     for filename, histogram in self.histograms_list]
        return distances

    def find_similar_images(self, target_image_path, num_neighbors=5):
        target_histogram = self.extract_histogram(target_image_path)

        distances = self.compare_histograms(target_histogram)

        # Exclude the target image itself from the list of similar images
        distances = [(filename, distance) for filename, distance in distances if filename != os.path.basename(target_image_path)]

        top_n_closest = sorted(distances, key=lambda x: x[1])[:num_neighbors]

        return top_n_closest

# Defining VGG16 class
class VGG16ImageSearch:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        self.filenames = [os.path.join(self.dataset_dir, name) for name in sorted(os.listdir(self.dataset_dir))]

    def extract_features(self, img_path):
        input_shape = (224, 224, 3)
        img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = self.model.predict(preprocessed_img, verbose=False)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(features)
        return normalized_features

    def find_similar_images(self, target_image_path, num_neighbors=5):
        target_features = self.extract_features(target_image_path)

        feature_list = [self.extract_features(filename) for filename in tqdm(self.filenames, desc="Extracting features")]

        neighbors = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='brute', metric='cosine').fit(feature_list)

        distances, indices = neighbors.kneighbors([target_features])

        similar_images = []
        for i in range(1, num_neighbors + 1):
            similar_images.append((os.path.basename(self.filenames[indices[0][i]]), distances[0][i]))

        return similar_images

# Function to save results to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["Filename", "Distance"])
    df["Distance"] = df["Distance"].round(2) 
    df.to_csv(filename, index=False)

###### 
# Main function
######
def main():
    """
    Main function to execute the script.
    """
    # Parse the arguments
    args = parse_arguments()

    target_image_path = args.target_image

    # Defining folder paths
    INPUT_FOLDER = os.path.join("in", "flowers")
    OUTPUT_FOLDER = os.path.join("out", f"results_{args.method}.csv")

    if args.method == "histogram":
        image_search = HistogramImageSearch(INPUT_FOLDER)
        similar_images = image_search.find_similar_images(target_image_path)
    elif args.method == "vgg":
        image_search = VGG16ImageSearch(INPUT_FOLDER)
        similar_images = image_search.find_similar_images(target_image_path)

    # Include the target image as the first entry in the CSV
    target_filename = os.path.basename(target_image_path)
    target_distance = 0.0
    data = [(target_filename, target_distance)]

    # Add other similar images
    data.extend(similar_images)

    save_to_csv(data, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()