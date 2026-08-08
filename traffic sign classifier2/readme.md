# Traffic Sign Classification using Enhanced CNN

This project implements an advanced Convolutional Neural Network (CNN) to classify traffic signs using TensorFlow and Keras, incorporating Batch Normalization for improved stability and training performance.

## Dataset
The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.
* **Classes:** 43 different traffic sign classes.
* **Structure:** Contains `Train`, `Test`, and `Meta` folders.

## Dependencies
Ensure you have the following libraries installed:
pip install numpy pandas matplotlib pillow scikit-learn tensorflow keras
Preprocessing
The images undergo the following pipeline before training:

Resizing: Standardized to 32 × 32 pixels.
Normalization: Pixel values scaled to [0, 1].
Label Encoding: One-hot encoded using to_categorical.

## Usage
Download and extract the GTSRB dataset into the directory.
Open traffic sign classifier2.ipynb in Jupyter Notebook.
Execute the cells sequentially.

## CNN Architecture & Model
This model utilizes a deep architecture with stacked convolutional layers and batch normalization to reduce internal covariate shift.

Block 1 (32 filters): Two Conv2D layers + BatchNormalization + ReLU + MaxPooling2D + Dropout.

Block 2 (64 filters): Two Conv2D layers + BatchNormalization + ReLU + MaxPooling2D + Dropout.

Block 3 (128 filters): Two Conv2D layers + BatchNormalization + ReLU + MaxPooling2D + Dropout.

Classifier: Flatten + Dense(512) + BatchNormalization + Dropout + Dense(43, Softmax).

## Model Summary

Layer (type)	Output Shape	Param #
InputLayer	(None, 32, 32, 3)	0
Block 1 (Conv2D x2)	(None, 32, 32, 32)	10,272
Block 2 (Conv2D x2)	(None, 16, 16, 64)	55,936
Block 3 (Conv2D x2)	(None, 8, 8, 128)	222,080
Dense (512)	(None, 512)	1,049,088
Output (Dense 43)	(None, 43)	22,059

## Training Configuration
* **Optimizer:** Adam (with learning rate scheduling, reduced to `1.2500e-04` by the end of training)
* **Loss Function:** Categorical Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 30
* **Batch Size:** 64 (approximately 350 steps per epoch)
* **Regularization:** Batch Normalization and Dropout layers applied systematically to prevent overfitting.

## Results
The enhanced architecture achieved excellent convergence and generalization on the GTSRB dataset:
* **Training Accuracy:** 100.00% (`loss: 0.0027`)
* **Validation Accuracy:** 99.97% (`val_loss: 0.0011`)
* **Test Accuracy:** 98.00% (`test_loss: 0.0576`)

The learning curves show that the combination of double-convolutional blocks, batch normalization, and learning rate adjustment successfully eliminated overfitting, pushing the test accuracy to **98%**.
Normalization Strategy: Batch Normalization applied after every Conv2D and the dense layer.
Regularization: Dropout applied after pooling and dense layers to prevent overfitting.
Results
