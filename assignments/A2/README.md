# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

This project provides a framework for comparing classification performance using Logistic Regression and Neural Networks on the CIFAR-10 dataset.

## Introduction

The aim of this project is to demonstrate the effectiveness of Logistic Regression and Neural Networks in classifying images from the CIFAR-10 dataset. It includes functionalities for data preprocessing, model training, evaluation, and result visualization.

## Getting Started

To run this project, ensure you have the necessary dependencies installed. You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Execute the `main.py` script to perform classification benchmarks with Logistic Regression and Neural Networks. You can specify the model type using the `--model` argument. Available options are `logistic` and `neural`.

```bash
python main.py --model logistic
```

or

```bash
python main.py --model neural
```

## Functionality

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

## Data

The CIFAR-10 dataset is used in this project. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Output

- Classification report: Saved as a text file containing precision, recall, and F1-score.
- Loss curve plot (for Neural Network): Saved as an image in the `out` folder.
