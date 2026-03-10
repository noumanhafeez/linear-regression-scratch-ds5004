import numpy as np
import pickle

class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.w = 0
        self.b = 0
        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.w * X + self.b
            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        return self.w * X + self.b

    def save_model(self, path="models/gradient_descent_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)