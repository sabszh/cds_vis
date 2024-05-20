#!/usr/bin/python
"""
Assignment 4 - Detecting faces in historical newspapers
Course: Visual Analytics
Author: Sabrina Zaki Hansen
"""

# Importing libraries
import os
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Allow loading truncated images to prevent errors when processing corrupted images
import pandas as pd
import matplotlib.pyplot as plt

######
# Defining functions
######

def extract_decade_from_filename(filename):
    """
    Extracts the decade from the given filename.

    Args:
        filename (str): The filename from which to extract the decade.

    Returns:
        int: The extracted decade.
    """
    parts = filename.split("-")
    year = int(parts[1])
    decade = (year // 10) * 10
    return decade

def process_files(folder_path):
    """
    Process all files in the specified folder and create a log file with the number of faces for each image.

    Args:
        folder_path (str): The path to the folder containing newspaper images.

    Returns:
        None
    """
    print("Processing files...")
    mtcnn = MTCNN(keep_all=True)
    face_counts = []

    for newspaper_folder in os.listdir(folder_path):
        full_path = os.path.join(folder_path, newspaper_folder)

        if os.path.isdir(full_path):
            newspaper_name = newspaper_folder.upper()
            print(f"Processing newspaper: {newspaper_name}")
            for filename in tqdm(sorted(os.listdir(full_path)), desc=newspaper_name):
                file_path = os.path.join(full_path, filename)
                if os.path.isfile(file_path) and filename.endswith(".jpg"):
                    try:
                        decade = extract_decade_from_filename(filename)
                        img = Image.open(file_path)
                        img = img.convert("RGB")
                        
                        # Resize image to speed up processing
                        img = img.resize((512, 512))
                        
                        boxes, _ = mtcnn.detect(img)
                        num_faces = len(boxes) if boxes is not None else 0

                        face_counts.append({'newspaper': newspaper_name, 'file': filename, 'num_faces': num_faces, 'decade': decade})
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    # Save face counts to a CSV file
    face_counts_df = pd.DataFrame(face_counts)
    os.makedirs('out', exist_ok=True)
    face_counts_df.to_csv('out/face_counts.csv', index=False)
    print(f"Face counts saved to 'out/face_counts.csv'")

def plot_percentage_of_faces_per_decade(face_counts_df):
    """
    Plot the percentage of pages with faces per decade for each newspaper and save the plot.

    Args:
        face_counts_df (pandas.DataFrame): DataFrame containing face counts.

    Returns:
        None
    """
    print("Plotting data...")
    # Calculate the total number of pages for each decade and newspaper
    total_pages_per_decade = face_counts_df.groupby(['decade', 'newspaper']).size().reset_index(name='total_pages')
    # Calculate the total number of pages with faces for each decade and newspaper
    pages_with_faces_per_decade = face_counts_df[face_counts_df['num_faces'] > 0].groupby(['decade', 'newspaper']).size().reset_index(name='pages_with_faces')
    # Merge the two dataframes
    merged_df = pd.merge(total_pages_per_decade, pages_with_faces_per_decade, on=['decade', 'newspaper'], how='left').fillna(0)
    # Calculate the percentage of pages with faces
    merged_df['percentage_pages_with_faces'] = (merged_df['pages_with_faces'] / merged_df['total_pages']) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    for newspaper, data in merged_df.groupby('newspaper'):
        plt.plot(data['decade'], data['percentage_pages_with_faces'], marker='o', linestyle='-', label=newspaper)
    plt.title('Percentage of Pages with Faces per Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Pages with Faces')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join("out", "percentage_of_faces_per_decade.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

###### 
# Main function
######

def main():
    folder_path = os.path.join("in", "newspapers")
    print(f"Starting processing for folder: {folder_path}")
    process_files(folder_path)
    face_counts_df = pd.read_csv('out/face_counts.csv')
    plot_percentage_of_faces_per_decade(face_counts_df)
    print("Processing complete.")

if __name__ == "__main__":
    main()
