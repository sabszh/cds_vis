# Assignment 3 - Document Classification using Pretrained Image Embeddings

This script is developed as part of Assignment 3 for Language Analytics course. The goal of this assignment is to classify documents based solely on their visual appearance, rather than their textual content. This is achieved by leveraging pretrained image embeddings and Convolutional Neural Networks (CNNs).

## Usage

To use this script, follow these steps:

1. Ensure you have all the necessary dependencies installed. You can install them using `pip install -r requirements.txt`.

2. Make sure you have access to the Tobacco3482 dataset. You can download it [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download)

3. Run the script using the following command:

   ```
   python script.py --data_path <path_to_image_data> --output_dir <output_directory_path>
   ```

   - `data_path`: Path to the image data directory.
   - `output_dir`: Directory where the results will be saved.

## Functionality

1. **Loading Data**: The script loads images from the specified data path. Each image is associated with a label indicating its document type.

2. **Preprocessing Images**: Images are preprocessed to prepare them for input into the VGG16 model. Preprocessing includes resizing, converting to arrays, and applying preprocessing specific to the VGG16 model.

3. **Building Model**: A CNN model based on VGG16 architecture is constructed with additional classification layers. The model is compiled using Adam optimizer and categorical crossentropy loss.

4. **Training the Model**: The model is trained on the preprocessed image data. Training is performed for a fixed number of epochs with a validation split. Training history including loss and accuracy curves are plotted and saved.

5. **Evaluation**: The trained model is evaluated using the test data. A classification report is generated and saved to assess the performance of the model.

## Results

### Classification Report

The classification report provides insights into the performance of the model. It includes metrics such as precision, recall, and F1-score for each class, along with overall accuracy.

### Learning Curves

Learning curves depict the training and validation loss, as well as training and validation accuracy over epochs. These curves help in understanding the model's convergence and potential overfitting or underfitting issues.

## Project Structure

The project should be structured as follows:

```
project_root/
│
├── src/
│   └── script.py               # Main script for document classification
│
├── data/
│   └── Tobacco3482/            # Directory containing image data
│
├── out/
│   └── classification_report.txt  # Classification report saved here
│   └── training_curves.png     # Plot of training curves
│
└── README.md                   # README file explaining the project and usage
```

## Notes

- Ensure that you have sufficient computational resources as running the script might be computationally intensive, especially during training.
- This script demonstrates the usage of TensorFlow for training CNNs and interpreting machine learning outputs.