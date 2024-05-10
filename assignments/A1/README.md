# Assignment 1: Building a Simple Image Search Algorithm
This script is designed to perform a simple image search algorithm based on color histograms or feature extraction using VGG16 with K-Nearest Neighbors (KNN). It compares the features or color histograms of a target image to those of other images in a dataset and identifies the most similar images.

## Data source
The dataset used for this asssignment can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). The data is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford.

## Requirements
- Python > 3.10.12
- OpenCV (`cv2`) library
- NumPy (`numpy`) library
- Pandas (`pandas`) library
- TensorFlow (`tensorflow`) library
- tqdm (`tqdm`) library

## Usage
To use this script, follow these steps:

1. Clone or download the repository and make sure you have the file structure as pointed out, and the needed files stored in `in`

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh <target_image> <--method>
    ```
    - `<target_image>`: Path to image to compare with, example: 
    - `<--method>`: Method for image search, either 'histogram' for histogram or 'vgg' for VGG using K-nearest-neighbor search.
    - Example: `bash run.sh in/flowers/image_1234.jpg --method vgg` 

## Script overview
This Python script is designed for conducting an image seach. It has two methods at use, either using a search based on histograms for colors represented in the image or a neural network approach. 
Below is an overview of the functions provided in the script:

### Histogram-based Image Search
- `extract_color_hist(image_path)`: Extracts color histograms for a single image.
- `compare_histograms(target_histogram, histograms_list)`: Compares the histogram of a target image to other histograms using Chi-Squared distance.

### VGG16 with KNN Image Search
- `extract_features(img_path, model)`: Extracts features from image data using the VGG16 model.
- `find_similar_images_vgg(target_image_path, dataset_dir, num_neighbors=5)`: Finds similar images to the target image using VGG16 features and KNN.

## Output Summary
The script saves the results to a CSV file in the output folder. The CSV file contains the filenames of the top similar images that are most similar to the target image along with their respective distances.

### Table 1: Histogram output

| Filename       | Distance |
|----------------|----------|
| image_0004.jpg | 0.0      |
| image_0973.jpg | 53.49    |
| image_0763.jpg | 55.86    |
| image_0981.jpg | 58.39    |
| image_1025.jpg | 58.87    |
| image_0721.jpg | 59.67    |

### Table 2: VGG output
| Filename       | Distance |
|----------------|----------|
| image_0004.jpg | 0.0      |
| image_0030.jpg | 0.16     |
| image_0019.jpg | 0.18     |
| image_0006.jpg | 0.19     |
| image_0038.jpg | 0.19     |

The table lists the distance measurements for a set of image files. The "Filename" column indicates the name of each image file, and the "Distance" column displays the corresponding distance measurement.

## Discussion of Limitations and Possible Steps to Improvement


## File Structure
The project directory should be structured as follows:

```
.
A1/
│
├── in/
│   └── flowers/
│       ├── image_0001.jpg
│       ├── image_0002.jpg
│       └── ...
├── out/
│   ├── 5top_VGG.csv
│   └── 5top_hist.csv
├── src/
│   └── image_search.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```