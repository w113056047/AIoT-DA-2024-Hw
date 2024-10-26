# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used for this project is from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

![demo](demo.gif)

## Note

This project was created using natural language programming with github copilot. The chat log is at [chat_log.md](chat.md).

## Problem Statement

Predict whether a passenger survived the Titanic disaster based on various features such as age, gender, class, etc.

## Strategy

1. **Data Preprocessing**: Handle missing values, encode categorical variables, and create new features.
2. **Feature Selection**: Use Recursive Feature Elimination (RFE) to select important features.
3. **Model Training**: Train a Logistic Regression model with hyperparameter tuning using GridSearchCV.
4. **Evaluation**: Evaluate the model using cross-validation and confusion matrix.
5. **Prediction**: Generate predictions on the test dataset for submission.

## License

This project is licensed under the MIT License.