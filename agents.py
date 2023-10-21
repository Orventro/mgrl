import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from datetime import datetime

import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from abc import abstractmethod

class Agent:

    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def load_history(self, history):
        pass

class PriceModel:

    def __init__(self, model: SARIMAX | None = None):
        pass
    
    def fit(self, df: pd.DataFrame):
        self.history = df
        self.model = SARIMAX(df, order=(5, 1, 1))
        self.model = self.model.fit(full_output=False, disp=False)
    
    def predict(self, current_time: datetime, current_price: float, n: int):
        self.history.loc[current_time] = current_price
        new_model = SARIMAX(self.history, order=(5, 1, 1))
        new_model = new_model.filter(self.model.params)
        return new_model.forecast(n).values

class MLAgent(Agent):

    def __init__(self, price_model: PriceModel | None =None):
        if price_model is None:
            self.price_model = PriceModel()
        else:
            self.price_model = price_model
        
        self.timestep = 0
        self.dt = 5/60
        self.hydro_cost = 0.06
        self.hydro_prod = 100
        self.hydro_penalty = 10
        self.hydro_amount = 1000
        self.ts_start = 0

        self.num_samples = 100
        self.max_price_quantile = 0.3
    
    def load_history(self, history):
        self.history = history.copy()
        # self.price_model.fit(self.history, "20220601")
    
    def reset(self, info: dict):
        self.ts_start = info['ts_start']
        self.price_model.fit(self.history.iloc[:self.ts_start])

    def predict(self, state):
        t, batt, price, sol, hydro = state
        ts_to_buy = int(np.ceil((1-hydro) * self.hydro_amount / self.hydro_prod / self.dt))
        ts_left = int(np.round(288 * (1 - t)))
        current_ts_global = self.ts_start+self.timestep
        self.timestep += 1
        if ts_left <= 1:
            return np.array([0, 0])

        row = self.history.iloc[current_ts_global]
        current_time, current_price = row.name, row.MW
        pred = self.price_model.predict(current_time, current_price, ts_left)
        # pred = self.history.iloc[current_ts_global:current_ts_global+ts_left].MW.values
        buy_price = np.sort(pred)[min(ts_to_buy, ts_left-1)] 
        top_price = np.quantile(pred, 0.5)
        action = np.array([0, 0])
        if (buy_price >= current_price) or (ts_left <= ts_to_buy) or (current_price < 0):
            action[0] = 1
        if ts_left > 20:
            if current_price > np.quantile(pred, 0.6):
                action[1] = -1
            if current_price < np.quantile(pred, 0.4):
                action[1] = 1
        if current_price < 0:
            action[1] = 1
        return action