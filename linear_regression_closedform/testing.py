import matplotlib.pyplot as plt
import numpy as np
from linear_closedform_model import LinearRegressionClosedForm


# Dataset
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Train model
model = LinearRegressionClosedForm()
model.fit(X, y)

# Convert to numpy
X_np = np.array(X)
print("X_np:", X_np)

# Generate predictions for training data
y_pred = model.predict(X_np)

# Plot original data points
plt.scatter(X, y, label="Actual Data")

# Plot regression line
plt.plot(X_np, y_pred, label="Regression Line")

# Labels and title
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")

# Legend
plt.legend()

# Save image
plt.savefig("linear_regression_plot_closedform.png")


# Show plot
plt.show()