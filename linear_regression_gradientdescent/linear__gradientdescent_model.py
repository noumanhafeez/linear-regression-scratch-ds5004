import numpy as np

class LinearRegressionGD:

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        n = len(X)

        self.w = 0
        self.b = 0

        for _ in range(self.epochs):

            y_pred = self.w * X + self.b

            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db


    def predict(self, X):
        X = np.array(X)
        return self.w * X + self.b
