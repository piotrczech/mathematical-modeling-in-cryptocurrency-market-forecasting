import numpy as np
from typing import Dict, Optional
from src.Preprocessing.target_scaler import TargetScaler


class MetricsService:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scaler: Optional[TargetScaler] = None) -> Dict[str, float]:
        if y_scaler is not None:
            try:
                y_true_orig = y_scaler.inverse_transform(y_true)
                y_pred_orig = y_scaler.inverse_transform(y_pred)
            except Exception:
                y_true_orig = y_true
                y_pred_orig = y_pred
        else:
            y_true_orig = y_true
            y_pred_orig = y_pred
        
        mse = np.mean((y_true_orig - y_pred_orig) ** 2)
        mae = np.mean(np.abs(y_true_orig - y_pred_orig))
        rmse = np.sqrt(mse)
        
        mask = y_true_orig != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
        else:
            mape = np.inf
        
        delta = 1.0
        residual = np.abs(y_true_orig - y_pred_orig)
        huber = np.where(
            residual <= delta,
            0.5 * residual ** 2,
            delta * residual - 0.5 * delta ** 2
        )
        huber_loss = np.mean(huber)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'HUBER': huber_loss
        }
    
    @staticmethod
    def calculate_single_metric(y_true: np.ndarray, y_pred: np.ndarray, 
                              metric_name: str = 'MSE', y_scaler: Optional[TargetScaler] = None) -> float:
        metrics = MetricsService.calculate_metrics(y_true, y_pred, y_scaler)
        return metrics.get(metric_name.upper(), metrics['MSE'])
    
    @staticmethod
    def format_metrics_table(metrics: Dict[str, float], title: str = "Metryki błędów") -> str:
        lines = [f"\n=== {title} ==="]
        for metric_name, value in metrics.items():
            if np.isinf(value):
                lines.append(f"{metric_name:>6}: N/A (division by zero)")
            else:
                lines.append(f"{metric_name:>6}: {value:.4f}")
        lines.append("=" * (len(title) + 8))
        return "\n".join(lines)
