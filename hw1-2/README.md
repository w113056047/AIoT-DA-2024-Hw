# Multiple Linear Regression with Lasso Feature Selection


> This is a homework program finished by natural language programming. Here is the chatgpt prompt [link](https://chatgpt.com/share/670d4754-3660-8013-a775-9fc8ce3f7c29) to the program.

This Streamlit app performs multiple linear regression using Lasso for feature selection, following the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology. The app allows you to fetch a dataset from the web, perform feature engineering using one-hot encoding for categorical variables, select the target variable, and run Lasso to determine the most relevant features. Finally, it displays the model’s performance and allows users to view random prediction results.

## Features

- **Data Source**: Enter a URL to fetch a CSV dataset from the web.
- **Feature Engineering**: Perform one-hot encoding on selected categorical columns.
- **Target Column Selection**: Choose the target column for prediction.
- **Lasso Regression**: Automatically selects relevant features for linear regression.
- **Model Evaluation**: Displays mean squared error (MSE) and R-squared metrics for model performance.
- **Random Predictions**: Allows the user to display a random subset of prediction results, with a refresh button for updating the random selection.

## Requirements

Make sure to install the following Python packages:

```bash
pip install streamlit scikit-learn pandas numpy requests
```

## Usage

1. Clone or download this repository.
2. Run the Streamlit app using the command:
   ```bash
   streamlit run main.py
   ```

3. In the app:
    * Enter the URL for your dataset (CSV format).
    * Select the categorical columns to one-hot encode.
    * Choose the target column for prediction.
    * The app will automatically apply Lasso regression for feature selection.
    * You can view model performance metrics (MSE and R-squared).
    * Use the dropdown to select the number of random predictions to display.
    * Press the Refresh Results button to get a new set of random predictions.

## Example

Here’s a sample workflow:

 1. Enter the URL of a dataset, such as https://example.com/data.csv.
 2. Select categorical columns like Gender, Country for one-hot encoding.
 3. Choose the target column for prediction, e.g., Price.
 4. View the selected features, model evaluation metrics, and random predictions.

## CRISP-DM Steps

1. Business Understanding: The app focuses on predicting a target variable (e.g., prices, sales) using linear regression.
2. Data Understanding: Users can explore the dataset through the app’s data overview.
3. Data Preparation: Feature engineering with one-hot encoding is applied to categorical variables.
4. Modeling: Lasso regression is used for feature selection, followed by linear regression for predictions.
5. Evaluation: Model performance is evaluated using mean squared error and R-squared.
6. Deployment: The results are displayed in an interactive interface with Streamlit.

## License

This project is licensed under the MIT License.