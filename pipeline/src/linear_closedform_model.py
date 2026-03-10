import numpy as np
import pickle

class LinearRegressionClosedForm:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean

    def predict(self, X):
        X = np.array(X)
        return self.w * X + self.b

    def save_model(self, path="models/closed_form_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)