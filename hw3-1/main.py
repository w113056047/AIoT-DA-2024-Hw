import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Generate 300 random data points between 0 and 1000
np.random.seed(0)  # for reproducibility
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)

# 2. Define the target variable y based on the rule provided
y = np.where((X < 500) | (X > 800), 0, 1).flatten()

# 3. Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# 4. Train SVM model with linear kernel
svm = SVC(kernel='rbf', probability=True)
svm.fit(X, y)

# Generate points to plot decision boundaries
X_test = np.linspace(0, 1000, 300).reshape(-1, 1)

# 5. Get predictions and probabilities for visualization
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1
svm_decision = svm.decision_function(X_test)         # Decision values for SVM

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 6. Plot Logistic Regression predictions
ax1.scatter(X, y, c=y, cmap='bwr', alpha=0.6, label="Data points")
ax1.plot(X_test, log_reg.predict(X_test), color="blue", linestyle="--", label="Prediction Boundary")
ax1.plot(X_test, log_reg_probs, color="red", label="Probability of Class 1")
ax1.set_title("Logistic Regression Predictions")
ax1.set_xlabel("X")
ax1.set_ylabel("y / Probability of Class 1")
ax1.legend()

# 7. Plot SVM predictions and decision boundary
ax2.scatter(X, y, c=y, cmap='bwr', alpha=0.6, label="Data points")
# SVM decision boundary: when decision function = 0
ax2.plot(X_test, (svm_decision > 0).astype(int), color="purple", linestyle="--", label="Prediction Boundary")
ax2.plot(X_test, svm_decision, color="green", label="Decision Function")
ax2.set_title("SVM Predictions with Decision Boundary")
ax2.set_xlabel("X")
ax2.set_ylabel("Decision Value / Predicted Class")
ax2.legend()

plt.tight_layout()
plt.show()