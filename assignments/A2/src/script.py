########
## Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
########

# Importing libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.datasets import cifar10

########
## Defining needed functions
########

# Argument parsing
def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Classification benchmarks with Logistic Regression and Neural Networks")
    parser.add_argument("--model",
                        choices=["logistic", "neural"],
                        default="logistic",
                        help="Choose classification model (logistic or neural)")
    return parser.parse_args()

# Grayscaling of training images
def grayscaler(image_array):
    """
    Convert color images to grayscale.

    Args:
        image_array (list): List of input color images.

    Returns:
        list: List of grayscale images.
    """
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in image_array]
    return gray_images

# Normalizing images
def normalizer(grayed_image):
    """
    Normalize grayscale images.

    Args:
        grayed_image (list): List of input grayscale images.

    Returns:
        list: List of normalized images.
    """
    norm_images = [cv2.normalize(grayed_image[i], grayed_image[i], 0, 1.0, cv2.NORM_MINMAX) for i in range(len(grayed_image))]
    return norm_images

# Flattening images
def flattener(grayed_image):
    """
    Flatten grayscale images.

    Args:
        grayed_image (list): List of input grayscale images.

    Returns:
        list: List of flattened images.
    """
    total_pixels = np.prod(grayed_image[0].shape)
    return [image.reshape(-1, total_pixels) for image in grayed_image]

# Renaming labels
def labeler(train, test):
    """
    Rename numerical labels to corresponding class names.

    Args:
        train (numpy.ndarray): Array of training labels.
        test (numpy.ndarray): Array of testing labels.

    Returns:
        tuple: Tuple containing renamed training labels and renamed testing labels.
    """
    # List of corresponding labels
    label_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
    # Converting the numerical values to the label name
    y_train = np.vectorize(label_map.get)(train.flatten())
    y_test = np.vectorize(label_map.get)(test.flatten())

    return y_train, y_test

# Preprocessing the images
def preprocessor(train_images, test_images):
    """
    Preprocess input images.

    Args:
        train_images (numpy.ndarray): Array of training images.
        test_images (numpy.ndarray): Array of testing images.

    Returns:
        tuple: Tuple containing processed training images and processed testing images.
    """
    processed_train_array = np.vstack(flattener(normalizer(grayscaler(train_images))))
    processed_test_array = np.vstack(flattener(normalizer(grayscaler(test_images))))

    return processed_train_array, processed_test_array

# Training and testing Logistic Regression classifier
def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    """
    Train and test Logistic Regression classifier.

    Args:
        X_train (numpy.ndarray): Array of training features.
        y_train (numpy.ndarray): Array of training labels.
        X_test (numpy.ndarray): Array of testing features.
        y_test (numpy.ndarray): Array of testing labels.

    Returns:
        tuple: Tuple containing trained classifier and classification report.
    """
    # Specifying the model and fitting it
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    
    # Predicting on test data
    y_pred = classifier.predict(X_test)
    
    # Making a classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    
    return classifier, classifier_metrics

# Training and testing Neural Network classifier
def neural_network_classifier(X_train, y_train, X_test, y_test):
    """
    Train and test Neural Network classifier.

    Args:
        X_train (numpy.ndarray): Array of training features.
        y_train (numpy.ndarray): Array of training labels.
        X_test (numpy.ndarray): Array of testing features.
        y_test (numpy.ndarray): Array of testing labels.

    Returns:
        tuple: Tuple containing trained classifier and classification report.
    """
    # Defining the model
    classifier = MLPClassifier(
        activation="logistic",
        hidden_layer_sizes=(20,),
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001
    )
    
    # Fitting the model (training)
    classifier.fit(X_train, y_train)
    
    # Predicting on test data
    y_pred = classifier.predict(X_test)
    
    # Making a classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    
    return classifier, classifier_metrics

# Function for saving reports
def save_report(report, report_name):
    """
    Save classification report to a text file.

    Args:
        report (str): Classification report.
        report_name (str): Name of the report file.
    """
    path = os.path.join("..", "out", f"{report_name}.txt")
    
    with open(path, "w") as report_file:
        report_file.write(report)

# Function for saving loss curve
def loss_curve(classifier):
    """
    Plot and save loss curve during training of Neural Network classifier.

    Args:
        classifier (MLPClassifier): Trained Neural Network classifier.
    """
    # Making the loss curve during training
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training on Cifar10 data", fontsize=16)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')

    # Saving plot in out folder
    path = os.path.join("..", "out", "loss_curve_nn.png")
    plt.savefig(path)
    plt.close()


########
## Main function
########

def main():
    # Parsing command-line arguments
    args = parse_arguments()

    # Loading data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocessing of the image data
    processed_train_array, processed_test_array = preprocessor(X_train, X_test)

    # Rename labels
    y_train_lab, y_test_lab = labeler(y_train, y_test)

    if args.model == "logistic":
        # Classification using logistic regression
        classifier, classifier_metrics = logistic_regression_classifier(processed_train_array, y_train_lab, processed_test_array, y_test_lab)
    elif args.model == "neural":
        # Classification using neural network
        classifier, classifier_metrics = neural_network_classifier(processed_train_array, y_train_lab, processed_test_array, y_test_lab)

    # Saving report
    save_report(classifier_metrics, f"{args.model}_report")

    # Saving loss curve plot for neural network
    if args.model == "neural":
        loss_curve(classifier)

if __name__=="__main__":
    main()