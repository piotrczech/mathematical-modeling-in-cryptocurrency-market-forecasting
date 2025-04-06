from datetime import datetime, timezone
import pytz
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Client:
    BASE_URL = "https://api.blockchain.info"

    def __init__(self):
        self.available_charts = [
            "avg-block-size",
            "n-transactions-per-block",
            "n-payments-per-block",
            "transactions-per-second", 
            "blocks-size",
            "hash-rate",
            "difficulty",
        ]
        self.warsaw_tz = pytz.timezone("Europe/Warsaw")  # Strefa czasowa Warszawy

    def get_chart_data(self, chart_name, start=None, timespan=None, rolling_average="24hours", data_format="json", sampled=None):
        """ Pobiera dane wykresu Blockchain Charts. """
        if chart_name not in self.available_charts:
            raise ValueError(f"'{chart_name}' nie jest dostępne. Możliwe wykresy: {', '.join(self.available_charts)}")

        url = f"{self.BASE_URL}/charts/{chart_name}"
        params = {"format": data_format, "metadata": 'false'}

        if start:
            params["start"] = start
        if timespan:
            params["timespan"] = timespan  # Dodajemy obsługę timespan
        if rolling_average:
            params["rollingAverage"] = rolling_average
        if sampled is not None:
            params["sampled"] = str(sampled).lower()

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()

            # Konwersja timestamp -> datetime w strefie czasowej Warszawy
            for point in data["values"]:
                dt_utc = datetime.fromtimestamp(point["x"], tz=timezone.utc)
                point["date"] = dt_utc.astimezone(self.warsaw_tz) 

            return data
        else:
            response.raise_for_status()

    def plot_basic_blockchain_data(data, title, ylabel):
        """ Rysuje wykres z poprawnym formatem daty na osi X """
        dates = [point["date"] for point in data["values"]]
        values = [point["y"] for point in data["values"]]

        plt.figure(figsize=(12, 6))
        plt.plot(mdates.date2num(dates), values, label=title, color='blue', linewidth=2)  # 👈 Konwersja na format Matplotlib
        plt.xlabel("Data")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # Poprawne formatowanie osi X
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()