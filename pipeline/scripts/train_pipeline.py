import matplotlib.pyplot as plt
from src.linear_closedform_model import LinearRegressionClosedForm
from src.linear_gradientdescent_model import LinearRegressionGD
from src.utils import load_data, split_data, evaluate_model

# Step 1: Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 2: Train Closed Form Model
cf_model = LinearRegressionClosedForm()
cf_model.fit(X_train, y_train)
y_pred_cf = cf_model.predict(X_test)
cf_mse = evaluate_model(y_test, y_pred_cf)
print("Closed Form Model MSE:", cf_mse)

# Step 3: Train Gradient Descent Model
gd_model = LinearRegressionGD(lr=0.0001, epochs=10000)
gd_model.fit(X_train, y_train)
y_pred_gd = gd_model.predict(X_test)
gd_mse = evaluate_model(y_test, y_pred_gd)
print("Gradient Descent Model MSE:", gd_mse)

# Step 4: Save models
cf_model.save_model()
gd_model.save_model()

# Step 5: Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, cf_model.predict(X), color="red", label="Closed Form Regression")
plt.plot(X, gd_model.predict(X), color="green", label="Gradient Descent Regression")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.legend()
plt.savefig("images/house_price_regression.png", dpi=300)
plt.show()