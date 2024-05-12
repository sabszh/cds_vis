# Assignment 1: Building a Simple Image Search Algorithm
This script is designed to perform a simple image search algorithm based on color histograms or feature extraction using VGG16 with K-Nearest Neighbors (KNN) on images of flowers. It compares the features or color histograms of a target image to those of other images in the flower dataset and identifies the most similar images.

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
    - `<--method>`: Method for image search, either '--method histogram' for histogram or '--method vgg' for VGG using K-nearest-neighbor search.
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
| image_1234.jpg | 0.0      |
| image_1125.jpg | 10.5     |
| image_0575.jpg | 11.24    |
| image_0727.jpg | 12.18    |
| image_1124.jpg | 12.25    |
| image_0174.jpg | 13.08    |

### Table 2: VGG output
| Filename       | Distance |
|----------------|----------|
| image_1234.jpg | 0.0      |
| image_1215.jpg | 0.09     |
| image_1207.jpg | 0.11     |
| image_1233.jpg | 0.11     |
| image_1211.jpg | 0.11     |
| image_1212.jpg | 0.12     |

The table lists the distance measurements for a set of image files. The "Filename" column indicates the name of each image file, and the "Distance" column displays the corresponding distance measurement.

## Discussion of Limitations and Possible Steps to Improvement
One potential limitation of this approach is the absence of a robust benchmarking method for evaluating image searches beyond visual inspection. Currently, the assessment was done by manually comparing retrieved images with the target image based on visual similarity. One strategy to address this limitation could involve comparing the performance of the implemented methods against established and reliable image search algorithms. Alternatively, utilizing a pre-labeled dataset for evaluation could provide a more structured approach; however, this would still be constrained by the chosen similarity metric, which can vary depending on factors such as color composition, object categories, and image aspect ratios. Consequently, the effectiveness of both methods is heavily bound by their respective extraction techniques. For instance, the histogram-based method may encounter difficulties with images sharing similar color distributions but differing in content. Moreover, since it operates as a black-box model, we lack insights into the specific features it extracts and their potential generalizability across different datasets.

From a computational standpoint, both methods utilized in this search approach demand significant resources, particularly when considering potential scalability issues as the dataset could expand. The current implementation may encounter challenges related to scalability, especially in terms of memory usage for feature extraction, particularly with constrained computational capabilities. With each script execution, the process involves extracting histograms/features and then identifying target images. To alleviate the time and energy consumption associated with this process, one strategy could involve embedding the images and subsequently querying against nearby vectors.

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