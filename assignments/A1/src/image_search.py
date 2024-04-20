#!/usr/bin/python

"""
Assignment 1 - Building a simple image search algorithm
"""
# Importing packages
import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse

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