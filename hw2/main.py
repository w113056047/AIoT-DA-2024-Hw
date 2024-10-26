import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
logging.info("Loading dataset...")
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

# Data Exploration
logging.info("Exploring data...")
print(train_data.info())
print(train_data.describe())

# Data Preparation
logging.info("Preparing data...")
# Handle missing values in training data
logging.info("Handling missing values in training data...")
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)
train_data.drop(columns=["Cabin"], inplace=True)

# Handle missing values in test data
logging.info("Handling missing values in test data...")
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
test_data.drop(columns=["Cabin"], inplace=True)

# Encode categorical variables
logging.info("Encoding categorical variables...")
train_data = pd.get_dummies(train_data, columns=["Sex", "Embarked"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked"], drop_first=True)

# Align test data with training data
logging.info("Aligning test data with training data...")
X = train_data.drop(columns=["Survived", "Name", "Ticket", "PassengerId"])
y = train_data["Survived"]
test_data = test_data[X.columns]

# Feature Selection
logging.info("Selecting features...")
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
test_data_rfe = rfe.transform(test_data)

# Split the data
logging.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_rfe, y, test_size=0.2, random_state=42
)

# Train the model
logging.info("Training the model...")
model.fit(X_train, y_train)

# Make predictions on validation set
logging.info("Making predictions on validation set...")
y_pred = model.predict(X_test)

# Display confusion matrix
logging.info("Displaying confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Make predictions on test data
logging.info("Making predictions on test data...")
test_predictions = model.predict(test_data_rfe)

# Generate submission file
logging.info("Generating submission file...")
submission = pd.DataFrame(
    {
        "PassengerId": pd.read_csv("./data/test.csv")["PassengerId"],
        "Survived": test_predictions,
    }
)
submission.to_csv("submission.csv", index=False)
logging.info("Submission file generated: submission.csv")
