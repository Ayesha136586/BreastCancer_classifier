# Breast Cancer Classification with a Simple Neural Network

This notebook builds a simple feedforward neural network (using TensorFlow/Keras) to classify breast tumors as **malignant** or **benign** based on the classic Wisconsin Breast Cancer dataset.

## Dataset

- Source: `sklearn.datasets.load_breast_cancer()` (built into scikit-learn, no external download needed)
- 569 samples, 30 numeric features (e.g. radius, texture, perimeter, area, smoothness, etc. — mean, standard error, and "worst" values for each)
- Target distribution: 357 Benign (label `1`), 212 Malignant (label `0`)
- No missing values

### Preprocessing
- Features and target separated (`X`, `Y`)
- Train/test split: 80/20 (455 train / 114 test), `random_state=2`
- Features standardized using `StandardScaler` (fit on training data, applied to both train and test)

## Model Architecture

A simple `Sequential` Keras model:
- `Flatten(input_shape=(30,))`
- `Dense(20, activation='relu')`
- `Dense(2, activation='sigmoid')` — output layer for 2 classes

- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Trained for 10 epochs with a 10% validation split from the training data

## Training Results

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|---|---|---|---|
| 1 | 0.447 | 0.717 | 0.555 |
| 5 | 0.934 | 0.957 | 0.198 |
| 10 | 0.958 | 0.957 | 0.122 |

Training accuracy and loss curves (accuracy vs. epoch, loss vs. epoch) are plotted for both training and validation data.

## Test Set Performance

**Test accuracy: 93.9%**

## Predictive System

The notebook includes a ready-to-use inference example: given a new patient's 30 feature values, it standardizes the input with the same fitted `scaler`, runs it through the trained model, and prints whether the tumor is predicted to be **Malignant** or **Benign**.

## Requirements

```
tensorflow
scikit-learn
numpy
pandas
matplotlib
```

## How to Run

1. Run all cells in order — the dataset loads directly from scikit-learn, no external files needed.
2. The notebook walks through EDA, preprocessing, model building, training, evaluation, and a sample prediction on new data.
3. To classify a new sample, replace the values in `input_data` (Cell 48) with your own 30 feature measurements in the same order as the dataset's feature columns.

## Notes

- This is a lightweight, from-scratch NN meant as an introductory deep learning exercise — for production use, consider cross-validation, hyperparameter tuning, and a held-out validation set that doesn't come from a random split of the training data.
- The model achieves strong accuracy (~94%) with very few epochs, which is expected given this dataset is small, well-separated, and commonly used for teaching binary classification.
