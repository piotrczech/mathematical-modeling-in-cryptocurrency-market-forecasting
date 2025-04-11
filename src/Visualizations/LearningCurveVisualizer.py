import matplotlib.pyplot as plt
import pandas as pd

class LearningCurveVisualizer:
    def __init__(self, history):
        self.history = history

    def visualize(self):
        pd.DataFrame(self.history.history).plot(figsize=(8, 5))

        plt.grid(True)
        plt.title("Krzywa uczenia: Średnia funkcja straty (loss) oraz val_loss dla zbioru walidacyjnego")
        plt.xlabel("Epoki")
        plt.ylabel("Funkcja straty")
        plt.show()
