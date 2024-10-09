import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set the title of the web app
st.title('Linear Regression Demo')

# Sidebar for user inputs
st.sidebar.header('Control Panel')

# User inputs
slope = st.sidebar.number_input('Slope (m)', value=2.5, step=0.1)
noise_scale = st.sidebar.number_input('Noise Scale', value=2.0, step=0.1)
num_points = st.sidebar.slider('Number of Points', min_value=10, max_value=200, value=100)

# Step 1: Generate random data
np.random.seed(0)  # For reproducibility
x = np.random.rand(num_points, 1) * 10  # Random points between 0 and 10
y = slope * x.flatten() + np.random.randn(num_points) * noise_scale  # Linear relation with noise

# Step 2: Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Step 3: Get the slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

# Calculate the regression line
regression_line = model.predict(x)

# Step 4: Calculate Mean Squared Error
mse = mean_squared_error(y, regression_line)

# Step 5: Print results
# st.write("Generated Data Points:")
# for xi, yi in zip(x.flatten(), y):
#     st.write(f"x: {xi:.2f}, y: {yi:.2f}")

st.write("Regression Line:")
st.write(f"y = {m:.2f}x + {b:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")

# Plotting the data and regression line
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', label='Data Points')
ax.plot(x, regression_line, color='red', label='Regression Line')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression Example')
ax.legend()

# Display the plot in the Streamlit app
st.pyplot(fig)
