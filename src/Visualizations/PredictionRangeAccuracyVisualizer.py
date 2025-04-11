import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class PredictionVisualizationService:
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions

    def visualize(self):
        n_points = len(self.y_test)
        x_main = np.arange(n_points)

        real_high = self.y_test[:, 0]
        real_low = self.y_test[:, 1]
        real_close = self.y_test[:, 2]
        pred_high = self.predictions[:, 0]
        pred_low = self.predictions[:, 1]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(x_main, real_high, label='Rzeczywiste high', color='green', alpha=0.6)
        ax.plot(x_main, real_low, label='Rzeczywiste low', color='green', linestyle='--', alpha=0.6)

        ax.plot(x_main, pred_high, label='Przewidywane high', color='red', alpha=0.6)
        ax.plot(x_main, pred_low, label='Przewidywane low', color='red', linestyle='--', alpha=0.6)

        ax.plot(x_main, real_close, label='Rzeczywiste close', color='black', linewidth=2, alpha=0.5)

        # Parametr: liczba podziałów dla każdego przedziału między punktami.
        res = 20  # im wyższa wartość, tym więcej małych prostokątów

        for i in range(n_points - 1):
            x0 = x_main[i]
            x1 = x_main[i+1]
            dx = (x1 - x0) / res  # szerokość małego prostokąta

            for j in range(res):
                t = (j + 0.5) / res

                ph = pred_high[i] + (pred_high[i+1] - pred_high[i]) * t
                pl = pred_low[i]  + (pred_low[i+1]  - pred_low[i])  * t
                rh = real_high[i] + (real_high[i+1] - real_high[i]) * t
                rl = real_low[i]  + (real_low[i+1]  - real_low[i])  * t

                x_rect = x0 + j * dx

                green_top = min(ph, rh)
                green_bottom = max(pl, rl)

                # Sprawdzenie, czy w tym subprzedziale jest jakieś przecięcie
                if green_top > green_bottom:
                    # Rysujemy zielony prostokąt – fragment, gdzie predykcja pokrywa się z zakresem rzeczywistym.
                    rect_green = Rectangle(
                        (x_rect, green_bottom), dx, green_top - green_bottom,
                        facecolor='green', alpha=0.3
                    )
                    ax.add_patch(rect_green)

                    # Górny fragment (czerwony) – jeśli górna część predykcji wykracza poza przecięcie
                    if ph > green_top:
                        rect_red_upper = Rectangle(
                            (x_rect, green_top), dx, ph - green_top,
                            facecolor='red', alpha=0.3
                        )
                        ax.add_patch(rect_red_upper)
                    # Dolny fragment (czerwony) – jeśli dolna część predykcji wykracza poniżej przecięcia
                    if pl < green_bottom:
                        rect_red_lower = Rectangle(
                            (x_rect, pl), dx, green_bottom - pl,
                            facecolor='red', alpha=0.3
                        )
                        ax.add_patch(rect_red_lower)
                else:
                    # Jeżeli nie ma przecięcia, całość przedziału predykcji rysujemy na czerwono
                    rect_red = Rectangle(
                        (x_rect, pl), dx, ph - pl,
                        facecolor='red', alpha=0.3
                    )
                    ax.add_patch(rect_red)

        ax.set_title("Zakresy trafione vs nietrafione – szczegółowe prostokąty")
        ax.set_xlabel("Indeks próbki (czas)")
        ax.set_ylabel("Wartość")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
