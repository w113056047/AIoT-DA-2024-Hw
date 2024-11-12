# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

# Streamlit app header
st.title("SVM Classifier with Adjustable Data Points and Boundary Conditions")
st.write("""
This app lets you control the number of random data points and the label boundary condition for a Support Vector Machine (SVM) classification.
""")

# Sidebar controls
n_points = st.sidebar.slider("Number of Data Points", min_value=50, max_value=1000, value=300, step=50)
boundary_limit = st.sidebar.slider("Label Boundary Condition (|x|, |y| < boundary)", min_value=100, max_value=1000, value=500, step=50)

# Step 1: Generate `n_points` random data points with two features (x, y)
np.random.seed(42)
x_coords = np.random.uniform(-1000, 1000, n_points)
y_coords = np.random.uniform(-1000, 1000, n_points)

# Step 2: Assign categories based on the selected boundary condition
labels = np.where((np.abs(x_coords) < boundary_limit) & (np.abs(y_coords) < boundary_limit), 0, 1)

# Combine x and y into a feature matrix
X = np.column_stack((x_coords, y_coords))
y = labels  # Target labels (0 for c1, 1 for c2)

# Step 3: Create and train an SVM model
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42))
model.fit(X, y)

# Step 4: Plot 2D data points with decision boundary
st.subheader("2D Plot of Data Points and SVM Decision Boundary")

# Create a grid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-1000, 1000, 500), np.linspace(-1000, 1000, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plotting in Streamlit
fig, ax = plt.subplots(figsize=(8, 6))

# Plot decision boundary and margins
ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=["#FFAAAA", "#AAAAFF", "#AAFFAA"])
ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.5)  # Decision boundary

# Plot data points for each category
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', edgecolor='k', label='Category c1')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', edgecolor='k', label='Category c2')

# Set plot labels and title
ax.set_xlabel('Feature x')
ax.set_ylabel('Feature y')
ax.set_title('2D SVM Decision Boundary and Data Points')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Step 5: 3D Visualization of the Decision Hyperplane

st.subheader("3D Plot of Data Points and SVM Decision Hyperplane")

# Extend the 2D data into 3D space by adding a third dimension (e.g., decision function)
Z_plane = model.decision_function(X)  # Get decision function values for each data point
X_3D = np.column_stack((x_coords, y_coords, Z_plane))  # Create a 3D version of data points

# Plot in 3D using Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each category with different colors
ax.scatter(X_3D[y == 0][:, 0], X_3D[y == 0][:, 1], X_3D[y == 0][:, 2], color='blue', edgecolor='k', label='Category c1')
ax.scatter(X_3D[y == 1][:, 0], X_3D[y == 1][:, 1], X_3D[y == 1][:, 2], color='red', edgecolor='k', label='Category c2')

# Set 3D plot labels and title
ax.set_xlabel('Feature x')
ax.set_ylabel('Feature y')
ax.set_zlabel('Decision Function')
ax.set_title('3D SVM Decision Hyperplane and Data Points')
ax.legend()

st.pyplot(fig)