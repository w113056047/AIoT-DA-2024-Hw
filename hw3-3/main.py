import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# 1. Business Understanding:
#   - Task: Classify non-linearly separable data using SVM
#   - Data: Synthetic dataset generated by make_moons

# 2. Data Understanding:
#   - Generate the dataset
X, y = make_moons(n_samples=1000, noise=0.15)

# 3. Data Preparation:
#   - Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Modeling:
#   - Create an SVM model
model = SVC(kernel="rbf", C=1.0)

#   - Train the model
model.fit(X_train, y_train)

# 5. Evaluation:
#   - Make predictions on the test set
y_pred = model.predict(X_test)

#   - Evaluate the model's performance (optional, e.g., using accuracy score)
#   from sklearn.metrics import accuracy_score
#   accuracy = accuracy_score(y_test, y_pred)
#   print("Accuracy:", accuracy)

# 6. Deployment:
#   - Visualize the results

# 2D Plot:
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

# Create a meshgrid for plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Predict the class labels for each point in the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis", edgecolor="k")
plt.title("SVM on Moon Dataset")
plt.show()

# 3D Plot:
# Add a third dimension by projecting the data onto a higher-dimensional space
X_3D = np.c_[X, X[:, 0] ** 2 + X[:, 1] ** 2]

# Create a 3D SVM model
model_3D = SVC(kernel="rbf")
model_3D.fit(X_3D, y)

# Create a 3D meshgrid
x1_min, x1_max = X_3D[:, 0].min() - 1, X_3D[:, 0].max() + 1
x2_min, x2_max = X_3D[:, 1].min() - 1, X_3D[:, 1].max() + 1
x3_min, x3_max = X_3D[:, 2].min() - 1, X_3D[:, 2].max() + 1
xx1, xx2, xx3 = np.meshgrid(
    np.arange(x1_min, x1_max, 0.2),
    np.arange(x2_min, x2_max, 0.2),
    np.arange(x3_min, x3_max, 0.2),
)

# Predict the class labels for each point in the 3D meshgrid
Z = model_3D.predict(np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()])
Z = Z.reshape(xx1.shape)

# 3D Plot:
# Add a third dimension by projecting the data onto a higher-dimensional space
X_3D = np.c_[X, X[:, 0] ** 2 + X[:, 1] ** 2]

# Create a 3D SVM model
model_3D = SVC(kernel="rbf")
model_3D.fit(X_3D, y)

# Create a 2D meshgrid for plotting the decision boundary
x1_min, x1_max = X_3D[:, 0].min() - 1, X_3D[:, 0].max() + 1
x2_min, x2_max = X_3D[:, 1].min() - 1, X_3D[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))

# Predict the class labels for each point in the 2D meshgrid
Z = model_3D.predict(np.c_[xx1.ravel(), xx2.ravel(), (xx1**2 + xx2**2).ravel()])
Z = Z.reshape(xx1.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the data points
ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=y, cmap="viridis")

# Plot the decision boundary
ax.plot_surface(xx1, xx2, Z, alpha=0.3)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
plt.title("3D SVM Decision Boundary")
plt.show()
