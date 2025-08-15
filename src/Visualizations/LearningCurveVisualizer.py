import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.Services.MetricsService import MetricsService
from src.Preprocessing.target_scaler import TargetScaler
from typing import Optional, Dict, List

class LearningCurveVisualizer:
    def __init__(self, history, loss_function_name: str = "huber", 
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 y_scaler: Optional[TargetScaler] = None, model=None):
        self.history = history
        self.loss_function_name = loss_function_name.upper()
        self.X_val = X_val
        self.y_val = y_val
        self.y_scaler = y_scaler
        self.model = model

    def visualize(self):
        """Wyświetla krzywą uczenia z wszystkimi metrykami."""
        if self._can_calculate_original_metrics():
            self._plot_full_metrics_layout()
        else:
            self._plot_simplified_layout()

    def _plot_full_metrics_layout(self):
        """Wyświetla pełny layout z osobnymi subplotami dla każdej metryki."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Krzywe uczenia - Funkcja treningu: {self.loss_function_name} (pogrubiona)", fontsize=16)
        
        # Panel 1: Oryginalne loss (scaled)
        self._plot_original_loss(axes[0, 0])
        
        # Oblicz metryki w oryginalnej skali
        epochs = len(self.history.history['loss'])
        y_pred = self.model.predict(self.X_val, verbose=0)
        final_metrics = MetricsService.calculate_metrics(self.y_val, y_pred, self.y_scaler)
        
        # Aproksymuj historię metryk
        loss_history = self.history.history.get('val_loss', self.history.history['loss'])
        final_loss = loss_history[-1]
        
        metrics_history = {}
        for epoch in range(epochs):
            loss_ratio = loss_history[epoch] / final_loss if final_loss > 0 else 1.0
            for metric_name, final_value in final_metrics.items():
                if metric_name not in metrics_history:
                    metrics_history[metric_name] = []
                metrics_history[metric_name].append(final_value * loss_ratio)
        
        # Panel 2-6: Osobne wykresy dla każdej metryki
        metric_positions = [
            (0, 1), (0, 2),  # Pierwszy rząd
            (1, 0), (1, 1), (1, 2)  # Drugi rząd
        ]
        
        metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE', 'HUBER']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i, metric_name in enumerate(metric_names):
            if i < len(metric_positions) and metric_name in metrics_history:
                row, col = metric_positions[i]
                ax = axes[row, col]
                
                # Pogrub linię jeśli to funkcja treningu
                line_width = 3 if metric_name == self.loss_function_name else 2
                alpha = 1.0 if metric_name == self.loss_function_name else 0.8
                
                ax.plot(metrics_history[metric_name], 
                       color=colors[i], linewidth=line_width, alpha=alpha,
                       label=f'{metric_name}' + (' (funkcja treningu)' if metric_name == self.loss_function_name else ''))
                
                ax.set_title(f'{metric_name}' + (' - Funkcja treningu' if metric_name == self.loss_function_name else ''), 
                           fontweight='bold' if metric_name == self.loss_function_name else 'normal')
                ax.set_xlabel("Epoki")
                ax.set_ylabel(f"Wartość {metric_name}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Dodaj wartość końcową jako tekst
                final_val = metrics_history[metric_name][-1]
                ax.text(0.95, 0.95, f'Końcowa: {final_val:.4f}', 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def _plot_simplified_layout(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        self._plot_original_metrics(ax)
        
        plt.tight_layout()
        plt.show()

    def _plot_original_loss(self, ax):
        history_df = pd.DataFrame(self.history.history)
        
        # Narysuj loss i val_loss
        if 'loss' in history_df.columns:
            ax.plot(history_df['loss'], label='Train Loss', 
                   linestyle='-', linewidth=3, alpha=0.8, color='blue')
        
        if 'val_loss' in history_df.columns:
            ax.plot(history_df['val_loss'], label='Val Loss', 
                   linestyle='--', linewidth=3, alpha=0.8, color='red')
        
        ax.set_title(f"Funkcja straty podczas treningu (scaled)\n{self.loss_function_name}")
        ax.set_xlabel("Epoki")
        ax.set_ylabel("Wartość funkcji straty")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Dodaj wartości końcowe
        if 'loss' in history_df.columns:
            final_train = history_df['loss'].iloc[-1]
            ax.text(0.05, 0.95, f'Końcowy Train: {final_train:.4f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        if 'val_loss' in history_df.columns:
            final_val = history_df['val_loss'].iloc[-1]
            ax.text(0.05, 0.85, f'Końcowy Val: {final_val:.4f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    def _plot_original_metrics(self, ax):
        """Wyświetla oryginalne metryki z historii treningu (scaled)."""
        history_df = pd.DataFrame(self.history.history)
        
        # Znajdź wszystkie metryki
        train_metrics = [col for col in history_df.columns if not col.startswith('val_')]
        val_metrics = [col for col in history_df.columns if col.startswith('val_')]
        
        # Narysuj wszystkie metryki
        for metric in train_metrics:
            line_style = '-'
            line_width = 3 if metric == 'loss' else 1.5
            ax.plot(history_df[metric], label=f'Train {metric}', 
                   linestyle=line_style, linewidth=line_width, alpha=0.8)
        
        for metric in val_metrics:
            line_style = '--'
            line_width = 3 if metric == 'val_loss' else 1.5
            ax.plot(history_df[metric], label=f'{metric}', 
                   linestyle=line_style, linewidth=line_width, alpha=0.8)
        
        ax.set_title(f"Metryki podczas treningu (scaled)\nFunkcja straty: {self.loss_function_name}")
        ax.set_xlabel("Epoki")
        ax.set_ylabel("Wartość metryki")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _can_calculate_original_metrics(self) -> bool:
        """Sprawdza czy można obliczyć metryki w oryginalnej skali."""
        return (self.X_val is not None and self.y_val is not None 
                and self.y_scaler is not None and self.model is not None)



    @staticmethod
    def create_enhanced_visualizer(history, loss_function_name: str = "huber",
                                 training_data: Optional[Dict] = None) -> 'LearningCurveVisualizer':
        if training_data:
            return LearningCurveVisualizer(
                history=history,
                loss_function_name=loss_function_name,
                X_val=training_data.get('X_val'),
                y_val=training_data.get('y_val'),
                y_scaler=training_data.get('y_scaler'),
                model=training_data.get('model')
            )
        else:
            return LearningCurveVisualizer(history, loss_function_name)
