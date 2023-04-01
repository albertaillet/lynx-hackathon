import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import evaluation

STRATEGIES = [
    'trend',
    'correlated_pairs',
    'optimization',
    'linreversion',
    'movingaverage',
]

SAVE_DIR = Path('position')


def get_prices() -> pd.DataFrame:
    return pd.read_csv('hackathon_prices_dev.csv', index_col='dates', parse_dates=['dates'])


def print_results(positions: pd.DataFrame, prices: pd.DataFrame) -> None:
    results = evaluation.calc_key_figures(positions=positions, prices=prices)
    # pad k with spaces to make it 10 chars long
    print(*[f'{k:15}: {v}' for k, v in results.items()], sep='\n')
    print()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for strategy in STRATEGIES:
        parser.add_argument(f'--{strategy}', action='store_true', help=f'Run {strategy} strategy')
    args = parser.parse_args()
    if not any([getattr(args, strategy) for strategy in STRATEGIES]):
        for strategy in STRATEGIES:
            setattr(args, strategy, True)
    return args


def normalize_positions(positions: pd.DataFrame) -> pd.DataFrame:
    return positions.div(positions.abs().sum(axis=1), axis=0)


def test_all_strategies() -> None:
    prices = get_prices()
    returns = prices.diff()

    args = get_args()

    total_pos = pd.DataFrame(0, index=returns.index, columns=returns.columns)

    if args.trend:
        from strategy_trend import trend_model

        # Test trend model
        trend_window = 250
        vol_window = 170
        filename = SAVE_DIR / f'trend_pos_{250=}_{170=}.csv'
        if filename.exists():
            trend_pos = pd.read_csv(filename, index_col='dates', parse_dates=['dates'])
        else:
            trend_pos = trend_model(ret=returns, trend_window=trend_window, vol_window=vol_window)
            trend_pos.to_csv(filename)
        total_pos += normalize_positions(trend_pos)
        print('Trend model:')
        print_results(positions=trend_pos, prices=prices)

    if args.correlated_pairs:
        from strategy_correlatedpairs import correlated_pairs_model

        # Test correlated pairs model
        lookback = 200
        vol_window = 100
        filename = SAVE_DIR / f'correlated_pairs_pos_{lookback=}_{vol_window=}.csv'
        if filename.exists():
            correlated_pairs_pos = pd.read_csv(filename, index_col='dates', parse_dates=['dates'])
        else:
            correlated_pairs_pos = correlated_pairs_model(
                prices=prices, lookback=lookback, vol_window=vol_window
            )
            correlated_pairs_pos.to_csv(filename)
        total_pos += normalize_positions(correlated_pairs_pos)
        print('Correlated pairs model:')
        print_results(positions=correlated_pairs_pos, prices=prices)

    if args.optimization:
        from strategy_optimization import optimization_model

        # Test sigma optimization model
        window = 5
        filename = SAVE_DIR / f'optimization_{window=}.csv'
        if Path(filename).exists():
            optimization_pos = pd.read_csv(filename, index_col='dates', parse_dates=['dates'])
        else:
            optimization_pos = optimization_model(ret=returns, window=window)
            optimization_pos.to_csv(filename)
        total_pos += normalize_positions(optimization_pos)
        print('Optimization model:')
        print_results(positions=optimization_pos, prices=prices)

    if args.linreversion:
        from strategy_linreversion import linreversion_model

        # Test linear reversion model
        rev_window = 100
        boundry1 = 0.05
        boundry2 = 0.025
        filename = SAVE_DIR / f'linreversion_{rev_window=}_{boundry1=}_{boundry2=}.csv'
        if Path(filename).exists():
            linreversion_pos = pd.read_csv(filename, index_col='dates', parse_dates=['dates'])
        else:
            linreversion_pos = linreversion_model(
                prices=prices, rev_window=100, boundry1=0.05, boundry2=0.025
            )
            linreversion_pos.to_csv(filename)
        total_pos += normalize_positions(linreversion_pos)
        print('Linear reversion model:')
        print_results(positions=linreversion_pos, prices=prices)

    if args.movingaverage:
        from strategy_movingaverage import movingaverage_model

        # Test moving average model
        T1 = 3
        T2 = 60
        vol_window = 100
        filename = SAVE_DIR / f'movingaverage_{T1=}_{T2=}_{vol_window=}.csv'
        if Path(filename).exists():
            movingaverage_pos = pd.read_csv(filename, index_col='dates', parse_dates=['dates'])
        else:
            movingaverage_pos = movingaverage_model(
                prices=prices, ret=returns, T1=T1, T2=T2, vol_window=vol_window
            )
            movingaverage_pos.to_csv(filename)
        total_pos += normalize_positions(movingaverage_pos)
        print('Moving average model:')
        print_results(positions=movingaverage_pos, prices=prices)

    print('--------------------------------')
    print('Total:')
    print_results(positions=total_pos, prices=prices)

    total_pos.to_csv(SAVE_DIR / 'total_pos.csv')

    evaluation.plot_key_figures(positions=total_pos, prices=prices)
    plt.show()


if __name__ == '__main__':
    test_all_strategies()
