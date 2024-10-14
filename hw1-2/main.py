import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import requests
from io import StringIO

# Step 1: Define data source (using a Streamlit text input for the URL)
st.title("Multiple Linear Regression using CRISP-DM")

url = st.text_input("Enter the URL of your dataset:", "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

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

    category_col = st.selectbox("Select the categorical column to encode:", data.columns)
    
    if category_col:
        encoder = OneHotEncoder()
        cat_encoded = encoder.fit_transform(data[[category_col]]).toarray()
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out([category_col]))
        
        # Drop the original category column and concatenate the encoded data
        data = pd.concat([data.drop(columns=[category_col]), cat_encoded_df], axis=1)
        
        st.write("Data after One-Hot Encoding:")
        st.write(data.head())

    # Step 3: Feature Selection (Simplified using correlation or any method)
    st.subheader("Feature Selection")
    target_col = st.selectbox("Select the target column for prediction:", data.columns)

    if target_col:
        # Remove target from the feature list
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Step 4: Model Building (Multiple Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        st.subheader("Model Evaluation")
        mse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared: {r2}")

        # Displaying the results
        st.write("Predictions vs Actual:")
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.write(result_df.head())

else:
    st.error("Please provide a valid data source URL.")