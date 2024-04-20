# Building a Simple Image Search Algorithm

This script is designed to perform a simple image search algorithm based on color histograms. It compares the color histograms of a target image to those of other images in a dataset and identifies the most similar images.

## Requirements
- Python 3
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pandas (`pandas`)

## Installation
1. Clone the repository: git clone <https://github.com/sabszh/cds_vis/tree/main/assignments/A1>

2. Install the required dependencies: bash setup.sh

## Usage
1. Ensure your images are stored in a folder.
2. Adjust the `INPUT_FOLDER` variable in the script to point to the folder containing your images.
3. Run the script: python script.py

4. Check the output folder for the CSV file containing the top 5 similar images to the target image.

## Functions
### `extract_color_hist(image_path)`
This function extracts color histograms for a single image.

- **Parameters:**
- `image_path` (str): Path to the image file.
- **Returns:**
- `numpy.ndarray`: Numpy array containing histograms for each channel (red, green, blue).

### `compare_histograms(target_histogram, histograms_list)`
Compares the histogram of a target image to other histograms using Chi-Squared distance.

- **Parameters:**
- `target_histogram` (numpy.ndarray): Histogram of the target image.
- `histograms_list` (list): List containing tuples of (filename, histograms) for all images.
- **Returns:**
- `list`: List of tuples where each tuple contains the filename and the Chi-Squared distance between the target image histogram and the histogram of each image in the dataset.

## Output
The script saves the results to a CSV file in the output folder. The CSV file contains the filenames of the top 5 images that are most similar to the target image along with their respective Chi-Squared distances.


## File Structure
The project directory should be structured as follows:

```
.
project_directory/
│
├── README.md
├── src/
│   ├── script.py
│   └── requirements.txt
│
├── in/
│   └── flowers/
│       ├── image_1.jpg
│       ├── image_2.jpg
│       └── ...
│
└── out/
    └── 5TOP_similar_images.csv 
```
