# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

This project provides a framework for comparing classification performance using Logistic Regression and Neural Networks on the CIFAR-10 dataset. The aim of this project is to demonstrate the effectiveness of Logistic Regression and Neural Networks in classifying images from the CIFAR-10 dataset. It includes functionalities for data preprocessing, model training, evaluation, and result visualization.

## Data Source
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Find more details [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Usage
To use this script, follow these steps:

1. Clone or download the repository and make sure you have the file structure as pointed out, and the needed files stored in `in`

2. Set up a virtual environment and install the required packages by running:
    ```
    bash setup.sh
    ```

3. Run the script by executing:
    ```
    bash run.sh <--model>
    ```
    - `<--model>`: Model for classification, either 'logistic' for logistic regression or 'neural' for neural network benchmarking.
    - Example: `bash run.sh --model neural` 

## Script Overview

- `parse_arguments()`: Parses command-line arguments.
- `grayscaler(image_array)`: Converts color images to grayscale.
- `normalizer(grayed_image)`: Normalizes grayscale images.
- `flattener(grayed_image)`: Flattens grayscale images.
- `labeler(train, test)`: Renames numerical labels to corresponding class names.
- `preprocessor(train_images, test_images)`: Preprocesses input images.
- `logistic_regression_classifier(X_train, y_train, X_test, y_test)`: Trains and tests Logistic Regression classifier.
- `neural_network_classifier(X_train, y_train, X_test, y_test)`: Trains and tests Neural Network classifier.
- `save_report(report, report_name)`: Saves classification report to a text file.
- `loss_curve(classifier)`: Plots and saves loss curve during training of Neural Network classifier.
- `main()`: Main function orchestrating the entire process.


## Output Summary
- Classification report: Saved as a text file containing precision, recall, and F1-score.
- Loss curve plot (for Neural Network): Saved as an image in the `out` folder.

## Discussion of Limitations and Possible Steps to Improvement


## File Structure
The project directory should be structured as follows:

```
.
A2/
│
├── out/
│   ├── logistic_report.txt
│   ├── loss_curve_nn.png
│   └── neural_report.txt
├── src/
│   └── classification.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```