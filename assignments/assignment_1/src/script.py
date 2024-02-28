########
## Assignment 1 - Building a simple image search algorithm
########

# Importing packages
import os
import cv2
import numpy as np
import pandas as pd

# Utility functions
import sys
sys.path.append(os.path.join(".."))

########
## Making functions
########

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

    # Split channels
    channels = cv2.split(image)

    histograms = []

    for channel in channels:
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [255], [0, 256])
        histograms.append(hist)

    # Concatenate histograms for all channels
    histograms = np.concatenate(histograms)

    # Normalize histogram
    histograms = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

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

    distances = []

    for filename, histogram in histograms_list:
        # Compute Chi-Squared distance
        distance = round(cv2.compareHist(target_histogram, histogram, cv2.HISTCMP_CHISQR),2)
        distances.append((filename, distance))

    return distances

########
## Running the functions and getting the comparisons
#######

# Defining folder path
folder_path = os.path.join("..","in", "flowers")

# Initialize an empty list to store (filename, histograms) tuples
histograms_list = []

# Iterate over each image in the folder
for image_filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_filename)

    # Call the function to extract color histograms for the current image
    histograms = extract_color_hist(image_path)

    # Append the filename and histograms tuple to the list
    histograms_list.append((image_filename, histograms))

# Defining image path for target image
image_path = os.path.join("..","in", "flowers","image_0555.jpg")

# Using function for extracting image histogram for a chosen image 
hist_chosen = extract_color_hist(image_path)

# Comparing histograms
distances = compare_histograms(hist_chosen,histograms_list)

# Sort the distances list based on the distances (second element of each tuple)
top_5_closest = sorted(distances, key=lambda x: x[1])[:5]

########
## Saving the dataframe as csv file
########

# Define the output CSV file path
results = pd.DataFrame(top_5_closest, columns = ["Filename", "Distance"])
results.to_csv(os.path.join("..", "out", "similar_images.csv"))

print(f"CSV file saved ")