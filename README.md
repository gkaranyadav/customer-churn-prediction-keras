
# Customer Churn Prediction

A machine learning project to analyze customer churn and predict customers who are likely to leave a telecom service.

## Problem Statement

Customer churn is an important problem for telecom companies. This project analyzes customer information and builds a binary classification model to identify customers who are more likely to churn.

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains 7,043 customer records and 21 columns. The target variable is `Churn`.

The application loads the public CSV automatically, so the dataset does not need to be stored in this repository.

## Project Workflow

- Load customer data
- Clean missing values
- Convert categorical features
- Perform basic churn analysis
- Scale input features
- Train a neural network
- Evaluate the model
- Display results using Streamlit

## Model

A neural network was built using TensorFlow and Keras.

The model contains:

- Dense layer with 64 neurons
- Dropout layer
- Dense layer with 32 neurons
- Dropout layer
- Sigmoid output layer

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Streamlit

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Run the Project

Clone the repository:

```bash
git clone https://github.com/gkaranyadav/customer-churn-prediction-keras.git
