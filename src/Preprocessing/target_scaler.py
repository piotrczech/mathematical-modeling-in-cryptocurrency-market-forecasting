import numpy as np
from sklearn.preprocessing import RobustScaler


class TargetScaler:
    """Skalowanie celu y (N, 2) z możliwością odwrotnej transformacji.

    Domyślnie RobustScaler, ale symetryczny (skala mediana/IQR) dobrze znosi outliery.
    """

    def __init__(self) -> None:
        self._scaler = RobustScaler()

    def fit(self, y_train: np.ndarray) -> "TargetScaler":
        if y_train.ndim != 2:
            raise ValueError(f"Expected 2D array (N,2), got {y_train.shape}")
        self._scaler.fit(y_train)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if y.ndim != 2:
            raise ValueError(f"Expected 2D array, got {y.shape}")
        return self._scaler.transform(y).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        if y_scaled.ndim != 2:
            raise ValueError(f"Expected 2D array, got {y_scaled.shape}")
        return self._scaler.inverse_transform(y_scaled)


