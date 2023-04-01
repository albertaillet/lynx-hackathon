import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def linreversion_model(prices: pd.DataFrame, rev_window=50, boundry1=0.05, boundry2=0.025):
    times = np.arange(rev_window).reshape(-1, 1)
    n_assets = prices.shape[1]
    # boundry = np.ones(n_assets) * boundry
    open = np.zeros(n_assets, dtype=bool)

    def minreversion(price_window, open, prev_pos, boundry1, boundry2):
        model = LinearRegression().fit(times[:-1], price_window[:-1])

        pred = model.predict(np.array([rev_window - 1]).reshape(-1, 1))
        actual = price_window[-1]
        diff = ((actual - pred) / actual).flatten()

        pos = np.zeros(n_assets)
        for i in range(n_assets):
            if open[i]:
                if abs(diff[i]) < boundry2:
                    open[i] = False
                else:
                    pos[i] = prev_pos[i]
            else:
                if abs(diff[i]) > boundry1:
                    pos[i] = -np.sign(diff[i])
                    open[i] = True
                else:
                    pass
        return pos, open

    pos = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for t in tqdm(range(rev_window + 1, prices.shape[0])):
        price_window = prices.iloc[t - rev_window : t].values
        prev_pos = pos.iloc[t - 1].values
        pos.iloc[t], open = minreversion(price_window, open, prev_pos, boundry1, boundry2)

    return pos
