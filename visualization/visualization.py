import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
data_path = "data/house_prices.csv"
df = pd.read_csv(data_path)

X = df["Size"].values
y = df["Price"].values

# -----------------------------
# Step 2: Load saved models
# -----------------------------
with open("models/closed_form_model.pkl", "rb") as f:
    cf_model = pickle.load(f)

with open("models/gradient_descent_model.pkl", "rb") as f:
    gd_model = pickle.load(f)

# -----------------------------
# Step 3: Generate predictions
# -----------------------------
y_pred_cf = cf_model.predict(X)
y_pred_gd = gd_model.predict(X)

# -----------------------------
# Step 4: Plot actual data and regression lines
# -----------------------------
plt.figure(figsize=(10, 6))

# Actual data points
plt.scatter(X, y, color="blue", label="Actual Data")

# Closed Form Regression Line
plt.plot(X, y_pred_cf, color="red", label="Closed Form Regression")

# Gradient Descent Regression Line
plt.plot(X, y_pred_gd, color="green", label="Gradient Descent Regression")

# Labels and Title
plt.xlabel("Size (sq ft)")
plt.ylabel("Price")
plt.title("House Price Prediction - Linear Regression")

# Legend
plt.legend()

# Save the figure
plt.savefig("images/house_price_regression.png", dpi=300)

# Show plot
plt.show()