import pandas as pd

class LoadDataFrameService:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df['real-daily-trend'] = df['monthly-btc-trend'] * df['daily-btc-trend'] / 100
        
        df.drop(columns=['monthly-btc-trend', 'daily-btc-trend'], inplace=True)

        return df
