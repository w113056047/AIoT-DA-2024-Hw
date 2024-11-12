# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Generate 300 random data points with two features (x, y)
# Initialize random seed for reproducibility
# np.random.seed(42)
n_points = 300

# Generate x and y coordinates from a uniform distribution between -1000 and 1000
x_coords = np.random.uniform(-1000, 1000, n_points)
y_coords = np.random.uniform(-1000, 1000, n_points)

# Step 2: Assign categories based on the given condition
# Category c1: |x| < 500 and |y| < 500; otherwise, category c2
labels = np.where((np.abs(x_coords) < 500) & (np.abs(y_coords) < 500), 0, 1)

# Combine x and y into a feature matrix
X = np.column_stack((x_coords, y_coords))
y = labels  # Target labels (0 for c1, 1 for c2)

# Step 3: Create and train an SVM model
# We use a rbf SVM to find a simple rbf decision boundary
model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
model.fit(X, y)

# Step 4: Plot the data points and the decision boundary

# Create a grid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-1000, 1000, 500), np.linspace(-1000, 1000, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 8))

# Plot decision boundary and margins
plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=["#FFAAAA", "#AAAAFF", "#AAFFAA"])
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.5)  # Decision boundary

# Plot data points for each category
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', edgecolor='k', label='Category c1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', edgecolor='k', label='Category c2')

# Set plot labels and title
plt.xlabel('Feature x')
plt.ylabel('Feature y')
plt.title('SVM Decision Boundary and Data Points')
plt.legend()
plt.grid(True)
plt.show()