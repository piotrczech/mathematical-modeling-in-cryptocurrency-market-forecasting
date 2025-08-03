import numpy as np
import matplotlib.pyplot as plt

class PredictionRangeEvaluatorVisualizer:
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions
        self.results = self._evaluate()
    
    def _evaluate(self):
        """Sprawdza, czy rzeczywista cena close mieści się w przewidywanym zakresie"""
        # Sprawdzamy i dopasowujemy kształty tablic
        if self.y_test.shape != self.predictions.shape:
            print(f"Uwaga: Różne rozmiary y_test ({self.y_test.shape[0]}) i predictions ({self.predictions.shape[0]}).")
            min_samples = min(self.y_test.shape[0], self.predictions.shape[0])
            print(f"Używam tylko pierwszych {min_samples} próbek.")
            self.y_test = self.y_test[:min_samples]
            self.predictions = self.predictions[:min_samples]
        
        # Wyciągamy potrzebne dane
        real_high = self.y_test[:, 0]
        real_low = self.y_test[:, 1]
        pred_high = self.predictions[:, 0]
        pred_low = self.predictions[:, 1]
        
        # Sprawdzamy, czy dane do ewaluacji mają prawidłowe wymiary
        assert len(real_high) == len(pred_high) == len(pred_low), "Niezgodne wymiary danych"
        
        # Liczymy ile razy cena close mieści się w przewidywanym zakresie
        in_range_high = (real_high <= pred_high) & (real_high >= pred_low)
        in_range_low = (real_low <= pred_high) & (real_low >= pred_low)
        accuracy = np.mean((in_range_high & in_range_low)) * 100
        
        # Średnia szerokość przewidywanego zakresu
        range_width = np.mean(pred_high - pred_low)
        
        # Dla przypadków poza zakresem, sprawdzamy o ile chybiliśmy
        out_range_mask = ~in_range_high & ~in_range_low
        above_high = np.sum(out_range_mask & (real_high > pred_high))
        below_low = np.sum(out_range_mask & (real_low < pred_low))
        
        # Średnie odchylenie dla przypadków poza zakresem
        if np.any(real_high > pred_high):
            deviation_above = np.mean((real_high[real_high > pred_high] - pred_high[real_high > pred_high]))
        else:
            deviation_above = 0
            
        if np.any(real_low < pred_low):
            deviation_below = np.mean((pred_low[real_low < pred_low] - real_low[real_low < pred_low]))
        else:
            deviation_below = 0
        
        return {
            'accuracy_percent': accuracy,
            'mae_high': np.mean(np.abs(real_high - pred_high)),
            'mae_low': np.mean(np.abs(real_low - pred_low)),
            'average_range_width': range_width,
            'count_above_high': above_high,
            'count_below_low': below_low,
            'avg_deviation_above': deviation_above,
            'avg_deviation_below': deviation_below
        }
    
    def print_summary(self):
        """Wyświetla podsumowanie oceny predykcji"""
        print(f"Dokładność predykcji zakresu: {self.results['accuracy_percent']:.2f}%")
        print(f"MAE dla high: {self.results['mae_high']:.2f}")
        print(f"MAE dla low: {self.results['mae_low']:.2f}")
        print(f"Średnia szerokość zakresu: {self.results['average_range_width']:.2f}")
        print(f"Liczba przypadków powyżej high: {self.results['count_above_high']}")
        print(f"Liczba przypadków poniżej low: {self.results['count_below_low']}")
        print(f"Średnie odchylenie powyżej high: {self.results['avg_deviation_above']:.2f}")
        print(f"Średnie odchylenie poniżej low: {self.results['avg_deviation_below']:.2f}")
    
    def plot_range_accuracy(self):
        """Wizualizuje dokładność predykcji zakresu"""
        real_high = self.y_test[:, 0]
        real_low = self.y_test[:, 1]
        pred_high = self.predictions[:, 0]
        pred_low = self.predictions[:, 1]
        
        # Tworzymy maskę dla trafień i chybień
        in_range_mask = ((real_high <= pred_high) & (real_high >= pred_low)) & ((real_low <= pred_high) & (real_low >= pred_low))
        
        plt.figure(figsize=(12, 6))
        
        # Rysujemy punkty dla trafień (zielone) i chybień (czerwone)
        x = np.arange(len(real_high))
        plt.scatter(x[in_range_mask], real_high[in_range_mask], c='green', alpha=0.6, label='High w zakresie')
        plt.scatter(x[~in_range_mask], real_high[~in_range_mask], c='red', alpha=0.6, label='High poza')
        plt.scatter(x[in_range_mask], real_low[in_range_mask], c='blue', alpha=0.6, label='Low w zakresie')
        plt.scatter(x[~in_range_mask], real_low[~in_range_mask], c='orange', alpha=0.6, label='Low poza')
        
        # Rysujemy zakresy
        for i in range(len(real_high)):
            plt.plot([i, i], [pred_low[i], pred_high[i]], 'k-', alpha=0.3)
        
        plt.title(f"Dokładność predykcji zakresu: {self.results['accuracy_percent']:.2f}%")
        plt.xlabel("Indeks próbki")
        plt.ylabel("Cena")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()