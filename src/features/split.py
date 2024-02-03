# class TimeSeriesSplit:
#     def __init__(self, n_splits=5, test_size=None) -> None:
#         self.n_splits=n_splits
#         self.test_size=test_size

#     def split(self, X):

import click
from sklearn.model_selection import TimeSeriesSplit

from src.io.utils import read_config, read_data_csv


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    config = read_config(config_path)
    resampled_data_path = config["data_paths"]["resampled_data_path"]

    chat_df = read_data_csv(resampled_data_path["chat_csv"], dates_cols=["time"])
    print(chat_df.head())
    X = chat_df.drop(["highlight"], axis=1)

    n_splits = 5
    gap = 6
    tscv = TimeSeriesSplit(n_splits, gap=gap)
    n_samples = X.shape[0]
    print(f"n_samples: {n_samples}")
    test_size = n_samples // (n_splits + 1)
    print(f"test_size: {test_size}")

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Split {i}")
        print(f"    Train: {train_index[0]}-{train_index[-1]}")
        print(f"    Length train: {len(train_index)}")
        print(f"    Test{test_index[0]}-{test_index[-1]}")
        print(f"    Length test: {len(test_index)}")

        print(X.loc[train_index[-1], "time"])
        print(X.loc[test_index[0], "time"])


if __name__ == "__main__":
    main()
