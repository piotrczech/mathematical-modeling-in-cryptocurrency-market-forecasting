import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class CryptoDataHandler:
    def __init__(self, path, sequence_length=30, test_days=21):
        self.path = path
        self.sequence_length = sequence_length
        self.test_days = test_days
        self.df = None
        self.X_full = None
        self.y_full = None

    def load_data(self):
        df = pd.read_csv(self.path)
        df['real-daily-trend'] = df['monthly-btc-trend'] * df['daily-btc-trend'] / 100
        df = df.drop(columns=['date', 'monthly-btc-trend', 'daily-btc-trend'])
        self.df = df
        return df

    def create_sequences(self):
        if self.df is None:
            self.load_data()
        df = self.df
        n_samples = len(df) - self.sequence_length
        n_features = df.shape[1]

        X = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float64)
        y = np.zeros((n_samples, 3), dtype=np.float64)

        for i in range(n_samples):
            X[i] = df.iloc[i:i+self.sequence_length].values
            y[i] = df.iloc[i+self.sequence_length][['high', 'low', 'close']].values

        self.X_full = X
        self.y_full = y
        return X, y
