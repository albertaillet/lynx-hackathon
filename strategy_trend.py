import numpy as np
import pandas as pd


def trend_model_unvectorized(ret, trend_window=50, vol_window=100):
    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    # loop over all dates
    for t in range(ret.shape[0]):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        vol = np.sqrt((ret**2).iloc[t - vol_window : t].mean())

        # Mean return between t-trend_window and t-1
        block_ret = ret.iloc[t - trend_window : t].sum()
        # Take a long position if the 50-days return is positive, otherwise take a short position (sign of the block return)
        unadj_pos = np.sign(block_ret)

        # Position at date t; risk adjust with volatility from previous date
        pos.iloc[t] = unadj_pos / vol
    return pos


def trend_model(ret, trend_window=250, vol_window=170):
    vol = np.sqrt((ret**2).rolling(window=vol_window).sum())
    pos_next_day = 1 / vol * np.sign(ret.rolling(window=trend_window).sum())
    pos = pos_next_day.shift(1)
    return pos
