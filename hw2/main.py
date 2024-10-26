import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
logging.info("Loading dataset...")
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
logging.info(f"Train data shape: {train_data.shape}")
logging.info(f"Test data shape: {test_data.shape}")

# Data Exploration
logging.info("Exploring data...")
logging.info(train_data.head())
logging.info(train_data.describe())

# Data Preparation
logging.info("Preparing data...")
# Handle missing values in training data
logging.info("Handling missing values in training data...")
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)
train_data.drop(columns=["Cabin"], inplace=True)
logging.info(train_data.isnull().sum())

# Handle missing values in test data
logging.info("Handling missing values in test data...")
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
test_data.drop(columns=["Cabin"], inplace=True)
logging.info(test_data.isnull().sum())

# Feature Engineering
logging.info("Creating new features...")
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
logging.info(train_data[["SibSp", "Parch", "FamilySize"]].head())

# Extract titles from names
train_data["Title"] = train_data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test_data["Title"] = test_data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# Simplify titles
title_mapping = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Dr": 5,
    "Rev": 6,
    "Col": 7,
    "Major": 7,
    "Mlle": 8,
    "Countess": 9,
    "Ms": 2,
    "Lady": 9,
    "Jonkheer": 10,
    "Don": 11,
    "Dona": 11,
    "Mme": 3,
    "Capt": 7,
    "Sir": 11,
}
train_data["Title"] = train_data["Title"].map(title_mapping)
test_data["Title"] = test_data["Title"].map(title_mapping)

# Fill missing titles with the most common one
train_data["Title"].fillna(1, inplace=True)
test_data["Title"].fillna(1, inplace=True)
logging.info(train_data[["Name", "Title"]].head())

# Encode categorical variables
logging.info("Encoding categorical variables...")
train_data = pd.get_dummies(train_data, columns=["Sex", "Embarked"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked"], drop_first=True)
logging.info(train_data.head())

# Align test data with training data
logging.info("Aligning test data with training data...")
X = train_data.drop(columns=["Survived", "Name", "Ticket", "PassengerId"])
y = train_data["Survived"]
test_data = test_data[X.columns]
logging.info(f"Features: {X.columns}")

# Feature Scaling
logging.info("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data)
logging.info(f"Scaled features sample: {X_scaled[:5]}")

# Feature Selection
logging.info("Selecting features...")
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_scaled, y)
test_data_rfe = rfe.transform(test_data_scaled)
logging.info(f"Selected features: {X.columns[rfe.support_]}")

# Hyperparameter Tuning
logging.info("Tuning hyperparameters...")
param_grid = {"C": [0.1, 1, 10, 100], "solver": ["liblinear", "saga"]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_rfe, y)
best_model = grid_search.best_estimator_
logging.info(f"Best hyperparameters: {grid_search.best_params_}")

# Cross-Validation
logging.info("Performing cross-validation...")
cv_scores = cross_val_score(best_model, X_rfe, y, cv=5)
logging.info(f"Cross-validation scores: {cv_scores}")
logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

# Split the data
logging.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_rfe, y, test_size=0.2, random_state=42
)

# Train the model
logging.info("Training the model...")
best_model.fit(X_train, y_train)

# Make predictions on validation set
logging.info("Making predictions on validation set...")
y_pred = best_model.predict(X_test)

# Display confusion matrix
logging.info("Displaying confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Make predictions on test data
logging.info("Making predictions on test data...")
test_predictions = best_model.predict(test_data_rfe)

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
