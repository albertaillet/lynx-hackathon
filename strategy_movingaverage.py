import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluation


def movingaverage_model(prices, ret, T1=3, T2=60, vol_window=100):
    pos = pd.DataFrame(0, index=ret.index, columns=ret.columns)
    for t in tqdm(range(ret.shape[0])):
        vol = np.sqrt((ret**2).iloc[t - vol_window : t].mean())

        moving_average1 = prices.iloc[t - T1 : t].mean()
        moving_average2 = prices.iloc[t - T2 : t].mean()
        unadj_pos = np.sign(moving_average1 - moving_average2)
        pos.iloc[t] = unadj_pos / vol

    return pos


if __name__ == '__main__':
    prices = pd.read_csv('hackathon_prices_dev.csv', index_col='dates', parse_dates=['dates'])
    # Compute returns
    ret = prices.ffill().diff()
    position = movingaverage_model(ret)
    print(evaluation.calc_key_figures(position, prices))
