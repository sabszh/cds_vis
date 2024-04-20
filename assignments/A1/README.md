# Assignment 1: Building a Simple Image Search Algorithm

This script is designed to perform a simple image search algorithm based on color histograms or feature extraction using VGG16 with K-Nearest Neighbors (KNN). It compares the features or color histograms of a target image to those of other images in a dataset and identifies the most similar images.

## Requirements
- Python 3
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- TensorFlow (`tensorflow`)
- tqdm (`tqdm`)

## Installation
1. Clone the repository: `git clone <https://github.com/sabszh/cds_vis/tree/main/assignments/A1>`
2. Navigate to the project directory: `cd A1`
3. Install the required dependencies: `bash setup.sh`

## Usage
1. Ensure your images are stored in a folder.
2. Run the script with the desired method:
   - For histogram-based search: `python script.py <path_to_target_image> <output_csv> --method histogram`
   - For VGG16 with KNN search: `python script.py <path_to_target_image> <output_csv> --method vgg`
3. Check the output folder for the CSV file containing the top similar images to the target image.

## Functions
### Histogram-based Image Search
- `extract_color_hist(image_path)`: Extracts color histograms for a single image.
- `compare_histograms(target_histogram, histograms_list)`: Compares the histogram of a target image to other histograms using Chi-Squared distance.

### VGG16 with KNN Image Search
- `extract_features(img_path, model)`: Extracts features from image data using the VGG16 model.
- `find_similar_images_vgg(target_image_path, dataset_dir, num_neighbors=5)`: Finds similar images to the target image using VGG16 features and KNN.

## Output
The script saves the results to a CSV file in the output folder. The CSV file contains the filenames of the top similar images that are most similar to the target image along with their respective distances.

## File Structure
The project directory should be structured as follows:

```
.
project_directory/
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
│   └── image_search.csv
│
├── README.md
├── requirements.txt
└── setup.sh
```