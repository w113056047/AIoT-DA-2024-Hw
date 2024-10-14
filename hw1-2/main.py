import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import requests
from io import StringIO

# Step 1: Define data source (using a Streamlit text input for the URL)
st.title("Multiple Linear Regression using CRISP-DM with Lasso Feature Selection")

url = st.text_input("Enter the URL of your dataset:", "https://example.com/data.csv")

@st.cache_data
def load_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            return data
        else:
            st.error(f"Failed to retrieve data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load data from the provided URL
data = load_data(url)

if data is not None:
    st.write("Data Overview:")
    st.write(data.head())

    # Step 2: Feature Engineering (One-Hot Encoding)
    st.subheader("Feature Engineering")

    # Select multiple categorical columns for encoding
    category_cols = st.multiselect("Select the categorical columns to encode:", data.columns)
    
    if category_cols:
        encoder = OneHotEncoder()
        cat_encoded = encoder.fit_transform(data[category_cols]).toarray()
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(category_cols))
        
        # Drop the original categorical columns and concatenate the encoded data
        data = pd.concat([data.drop(columns=category_cols), cat_encoded_df], axis=1)
        
        st.write("Data after One-Hot Encoding:")
        st.write(data.head())

    # Step 3: Target Column Selection
    st.subheader("Select the Target Column for Prediction")
    target_col = st.selectbox("Select the target column:", data.columns)

    if target_col:
        # Remove the target column from the feature set
        target = data[target_col]
        features = data.drop(columns=[target_col])

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Step 4: Feature Selection using Lasso Regression
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(X_train, y_train)
        
        # Get the selected features (non-zero coefficients)
        selected_features = features.columns[lasso_model.coef_ != 0]
        
        st.write(f"Selected Features by Lasso: {list(selected_features)}")

        # Step 5: Use the selected features for prediction with Linear Regression
        if len(selected_features) > 0:
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            model = LinearRegression()
            model.fit(X_train_selected, y_train)

            # Predict on the test set using selected features
            y_pred = model.predict(X_test_selected)

            # Evaluation Metrics (using mean squared error)
            st.subheader("Model Evaluation")
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test_selected, y_test)

            st.write(f"Mean Squared Error (MSE): {mse}")
            st.write(f"R-squared: {r2}")

            # Displaying the results
            st.write("Predictions vs Actual:")
            result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write(result_df.head())

else:
    st.error("Please provide a valid data source URL.")