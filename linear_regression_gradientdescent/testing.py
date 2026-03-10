import matplotlib.pyplot as plt
import numpy as np

from linear_regression_gradientdescent.linear__gradientdescent_model import LinearRegressionGD


# Dataset
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Train model
model = LinearRegressionGD(lr=0.01, epochs=1000)
model.fit(X, y)

print("Weight:", model.w)
print("Bias:", model.b)

# Convert to numpy for plotting
X_np = np.array(X)

# Generate predictions
y_pred = model.predict(X_np)

# Plot actual data
plt.scatter(X, y, color="blue", label="Actual Data")

# Plot regression line
plt.plot(X_np, y_pred, color="red", label="Regression Line")

# Plot predicted new point
new_x = [6]
new_y = model.predict(new_x)

plt.scatter(new_x, new_y, color="green", label="Prediction (x=6)")

# Labels
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression using Gradient Descent")

# Legend
plt.legend()

# Save image
plt.savefig("gradient_descent_regression.png", dpi=300)

# Show plot
plt.show()