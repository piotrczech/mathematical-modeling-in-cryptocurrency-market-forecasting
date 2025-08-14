import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

class DataProcessService:
    def __init__(
        self,
        sequence_length: int,
        test_days: int = 90,
        n_splits: int = 5,
    ):
        self.sequence_length = sequence_length
        self.test_days = test_days
        self.n_splits = n_splits

    def create_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df['date'] = pd.to_datetime(df['date'])
        dates = df['date'].values
        df_copy = df.drop(columns=['date'])

        max_valid_idx = len(df_copy) - self.sequence_length - 1  # Predict only 1 day ahead
        n_features = df_copy.shape[1]

        X_full = np.zeros((max_valid_idx, self.sequence_length, n_features), dtype=np.float32)
        y_full = np.zeros((max_valid_idx, 2), dtype=np.float32)  # Predicting [high, low]
        full_dates = np.zeros(max_valid_idx, dtype=object)

        for i in range(max_valid_idx):
            X_full[i] = df_copy.iloc[i : i + self.sequence_length].values
            prediction_idx = i + self.sequence_length
            y_full[i, 0] = df_copy.iloc[prediction_idx]['high']
            y_full[i, 1] = df_copy.iloc[prediction_idx]['low']
            full_dates[i] = dates[prediction_idx]

        return X_full, y_full, full_dates

    def split_data(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        full_dates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test = X_full[-self.test_days:]
        y_test = y_full[-self.test_days:]

        X_trainval = X_full[:-self.test_days]
        y_trainval = y_full[:-self.test_days]

        return X_trainval, y_trainval, X_test, y_test

    def get_time_series_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generator yielding (X_train, y_train, X_val, y_val) splits using TimeSeriesSplit.
        Ensures temporal consistency and avoids data leakage.
        """
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=1)
        for train_idx, val_idx in tss.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            yield X_train, y_train, X_val, y_val
