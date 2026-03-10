import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(path="data/house_prices.csv"):
    df = pd.read_csv(path)
    X = df["Size"].values
    y = df["Price"].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse