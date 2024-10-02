# Hackathon_team13

# Kannada OCR and Writer Identification

This project aims to develop a system for Optical Character Recognition (OCR) and writer identification for Kannada scripts.

## Overview

The system consists of two main components:

1. **Kannada OCR**: A deep learning-based model that can recognize Kannada characters from images.
2. **Writer Identification**: A clustering-based approach that identifies different writers based on their handwriting style.

## Features

### Kannada OCR
- Supports recognition of Kannada characters from images.
- Utilizes a convolutional neural network (CNN) architecture.
- Trained on a dataset of Kannada characters, ensuring a comprehensive understanding of the script.

### Writer Identification
- Employs K-Means clustering to identify different writers.
- Calculates features such as stroke width and character length to distinguish handwriting styles.
- Identifies writers based on unique characteristics of their handwriting.

## OCR Model Training

The OCR model undergoes several key stages during training:

### 1. Data Preprocessing
- **Image Segmentation**: The input images are processed to isolate individual characters using contour detection.
- **Normalization**: Images are converted to grayscale and normalized to ensure consistent input sizes and scales.
- **Augmentation**: Various transformations (like rotation, zoom, and shifts) are applied to increase dataset diversity and improve model robustness.

### 2. Model Architecture
The CNN architecture is designed to effectively learn and recognize features of Kannada characters through multiple convolutional layers.

#### CNN Layers Breakdown:
- **Convolutional Layers**: These layers apply filters to the input image, detecting edges, textures, and shapes. Each filter extracts features, which the model uses to learn representations of the characters.
  
- **Activation Function**: The ReLU (Rectified Linear Unit) activation function is applied after convolutional layers to introduce non-linearity, allowing the network to learn complex patterns.

- **Pooling Layers**: Max pooling is utilized to downsample the feature maps, reducing dimensionality while retaining important features. This also helps in achieving translational invariance.

- **Dropout Layers**: These layers randomly drop a percentage of neurons during training, helping to prevent overfitting by ensuring that the model doesn't become overly reliant on any single neuron.

- **Fully Connected Layers**: The output from the final convolutional layer is flattened and passed through fully connected layers to compute class probabilities for the character predictions.

### 3. Model Compilation
The model is compiled with the Adam optimizer, using categorical cross-entropy as the loss function, suitable for multi-class classification problems. 

### 4. Model Training
During training, the model learns to minimize the loss function through backpropagation, updating the weights based on the gradient of the loss. The training process is monitored using validation data to prevent overfitting, applying techniques such as early stopping and learning rate reduction when necessary.

# Writer Identification

The Writer Identification component of this project aims to distinguish different writers based on the characteristics of their handwriting. This is achieved through feature extraction and clustering techniques.

## Overview

Writer identification utilizes handwriting samples to identify distinct writing styles. By analyzing the features of handwritten text, the system clusters similar writing styles, allowing for accurate identification of individual writers.

## Features

- **Clustering-Based Identification**: 
  - Employs K-Means clustering to group handwriting samples by similarity.
  
- **Feature Extraction**:
  - Calculates key features from handwriting segments, such as:
    - **Stroke Width**: Measures the average width of the strokes in the handwriting.
    - **Character Length**: Assesses the length of individual characters to identify stylistic differences.
    
- **Handwriting Segmentation**:
  - Extracts individual segments of handwriting from input images, making it easier to analyze distinct samples.

## Methodology

### 1. Handwriting Segment Extraction
- The handwriting segments are extracted from a PDF file containing handwritten text.
- Each segment is preprocessed to enhance contrast and clarity, making it suitable for analysis.

### 2. Feature Calculation
- For each handwriting segment, the following features are calculated:
  - **Average Stroke Width**: This involves thresholding the image to obtain a binary representation and finding the contours of the strokes. The width of each stroke is measured, and the average is computed.
  - **Character Length**: The total number of characters in a segment is counted to help differentiate between writers.

### 3. Clustering
- The features from each segment are compiled into a feature vector.
- K-Means clustering is applied to these vectors to group similar handwriting samples together. The algorithm identifies clusters based on the calculated features, enabling the system to classify handwriting by writer.


## License

This project is licensed under the MIT License.
