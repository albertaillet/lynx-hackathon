import numpy as np
import pandas as pd
import scipy.optimize as opt
from tqdm import tqdm


def optimization_model(ret: pd.DataFrame, window=50):
    def optimize(rets, x0):
        C = np.cov(rets.T)
        mu = rets.mean(axis=0)

        def obj(x):
            return x.T @ C @ x

        x0 = np.ones(rets.shape[1]) / rets.shape[1]

        constraints = [
            {"type": "ineq", "fun": lambda x: np.dot(x, mu) - 0.0001},
        ]

        return opt.minimize(obj, x0, constraints=constraints).x

    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

    n_assets = ret.shape[1]
    x0_prev = np.ones(n_assets) / n_assets
    for t in tqdm(range(window + 1, ret.shape[0])):
        window_rets = ret.iloc[t - window : t]

        x0_prev = optimize(window_rets, x0_prev)
        pos.iloc[t] = x0_prev
    return pos
