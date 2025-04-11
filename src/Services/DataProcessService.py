import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessService:
    def __init__(self, sequence_length: int, test_days: int):
        self.sequence_length = sequence_length
        self.test_days = test_days

    def create_sequences(self, df) -> tuple[np.ndarray, np.ndarray]:
        df_copy = df.drop(columns=['date'])

        n_samples = len(df_copy) - self.sequence_length
        n_features = df_copy.shape[1]

        X_full = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float64)
        y_full = np.zeros((n_samples, 3), dtype=np.float64)

        for i in range(n_samples):
            X_full[i] = df_copy.iloc[i : i+self.sequence_length].values
            y_full[i] = df_copy.iloc[i+self.sequence_length][['high', 'low', 'close']].values
        
        return X_full, y_full

    def split_data(self, X_full: np.ndarray, y_full: np.ndarray):
        X_temp, X_test = X_full[:-(self.test_days + 1)], X_full[-(self.test_days + 1):]
        y_temp, y_test = y_full[:-(self.test_days + 1)], y_full[-(self.test_days + 1):]
        
        X_val, X_train, y_val, y_train = train_test_split(X_temp, y_temp, test_size=0.9, shuffle=True)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
