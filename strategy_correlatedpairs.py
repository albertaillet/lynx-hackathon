import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluation


def get_correlated_cluster(prices):
    corr = prices.corr()
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr = corr.stack().reset_index()
    corr.columns = ['var1', 'var2', 'corr']

    # drop correlations lower than 0.9
    corr = corr[corr['corr'] > 0.9]

    # sort by corr
    corr = corr.sort_values(by='corr', ascending=False)

    # create a set from all var1 and var2
    correlated_cluster = set(corr['var1'].unique()).union(set(corr['var2'].unique()))
    return correlated_cluster


def correlated_pairs_positions(t, prices, ret_window):
    correlated_cluster = get_correlated_cluster(prices)
    unadj_pos = pd.Series(0, index=prices.columns)
    ret = prices.ffill().diff()

    recent_logs = dict()
    for asset in correlated_cluster:
        recent_logs[asset] = ret[asset][t - ret_window - 1 : t - 1].mean()
    recent_avg = np.mean(list(recent_logs.values()))

    for asset in correlated_cluster:
        long_short = np.sign(recent_logs[asset] - recent_avg)
        unadj_pos[asset] = long_short * (recent_logs[asset] - recent_avg) / prices[asset][t - 1]

    return unadj_pos


def correlated_pairs_model(prices, lookback=200, vol_window=100):
    ret = prices.ffill().diff()
    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

    # loop over all dates
    for t in tqdm(range(ret.shape[0])):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        vol = np.sqrt((ret**2).iloc[t - vol_window : t].mean())

        unadj_pos = correlated_pairs_positions(t, prices, ret_window=lookback)

        # Position at date t; risk adjust with volatility from previous date
        pos.iloc[t] = unadj_pos / vol

    return pos


if __name__ == "__main__":
    prices = pd.read_csv('hackathon_prices_dev.csv', index_col='dates', parse_dates=['dates'])
    position = correlated_pairs_model(prices, lookback=200)
    print(evaluation.calc_key_figures(position, prices))
