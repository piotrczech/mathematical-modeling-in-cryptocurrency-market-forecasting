from pytrends.request import TrendReq

class Client:
    def __init__(self, keywords):
        self.keywords = keywords

    def get_chart_data(self, timeframe, geo='', gprop=''):
        """
        Pobiera dane z Google Trends dla podanej listy słów kluczowych w określonym przedziale czasowym.
        
        Parametry:
        - timeframe: przedział czasowy w formacie "YYYY-MM-DD YYYY-MM-DD" (np. "2015-05-02 2017-04-30")
        - geo: kod geograficzny (domyślnie pusty, czyli globalnie)
        - gprop: właściwość, np. 'news', 'images'; domyślnie pusty
        
        Zwraca:
        - DataFrame z danymi z Google Trends.
        """
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(self.keywords, timeframe=timeframe, geo=geo, gprop=gprop)
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df.drop(columns=["isPartial"], inplace=True)
        return df