import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class ForecastEngine:
    """
    Multi-Step Time-Series Forecast Engine
    Generates rolling lag features and trains gradient boosted regression trees.
    """
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100)

    def generate_lags(self, series, lags=3):
        df = pd.DataFrame(series)
        columns = [df.shift(i) for i in range(lags + 1)]
        df = pd.concat(columns, axis=1)
        df.dropna(inplace=True)
        return df.iloc[:, 1:], df.iloc[:, 0]

    def fit_and_predict(self, series):
        X, y = self.generate_lags(series)
        self.model.fit(X, y)
        last_lag = np.array([series[-3:]])
        return self.model.predict(last_lag)[0]

if __name__ == "__main__":
    series = np.array([10.0, 11.5, 12.0, 13.5, 15.0, 16.2, 17.5])
    forecaster = ForecastEngine()
    print("Forecasting Next Timestep Value:")
    print(forecaster.fit_and_predict(series))
