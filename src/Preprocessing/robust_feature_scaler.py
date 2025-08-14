import numpy as np
from sklearn.preprocessing import RobustScaler


class RobustFeatureScaler:
    """Prosty wrapper na RobustScaler dla tensora (N, T, F).

    Zasada: flatten po osi czasu na czas fit/transform, a potem z powrotem do (N, T, F).
    """

    def __init__(self) -> None:
        self._scaler = RobustScaler()

    def fit(self, X_train: np.ndarray) -> "RobustFeatureScaler":
        if X_train.ndim != 3:
            raise ValueError(f"Expected 3D array (N,T,F), got {X_train.shape}")
        X2d = X_train.reshape(-1, X_train.shape[-1])
        self._scaler.fit(X2d)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N,T,F), got {X.shape}")
        X2d = X.reshape(-1, X.shape[-1])
        Xt = self._scaler.transform(X2d)
        return Xt.reshape(X.shape).astype(np.float32)