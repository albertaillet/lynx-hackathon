# %%
import numpy as np
import pandas as pd
import scipy.optimize as opt
from tqdm import tqdm

import evaluation

# %%
prices = pd.read_csv('hackathon_prices_dev.csv', index_col='dates', parse_dates=['dates'])

# Compute returns
ret = prices.ffill().diff()


# %% Compute positions
def sigma_optimization_model(ret: pd.DataFrame, window=50):
    def optimize(rets, x0):
        C = np.cov(rets.T)
        mu = rets.mean(axis=0)

        def obj(x):
            return x.T @ C @ x - mu.T @ x

        x0 = np.ones(rets.shape[1]) / rets.shape[1]

        x = opt.minimize(obj, x0).x
        return x

    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

    n_assets = ret.shape[1]
    x0_prev = np.ones(n_assets) / n_assets
    for t in tqdm(range(window + 1, ret.shape[0])):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        window_rets = ret.iloc[t - window : t]

        # Position at date t; risk adjust with volatility from previous date
        x0_prev = optimize(window_rets, x0_prev)
        pos.iloc[t] = x0_prev
    return pos


pos = trend_model(ret, window=10)

# %% Evaluate strategy
evaluation.plot_key_figures(pos, prices)

# %%
