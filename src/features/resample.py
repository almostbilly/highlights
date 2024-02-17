import logging
import random
from typing import List, Tuple

import click
import numpy as np
import pandas as pd

from src.features.transformers import ChatTransformer, ClipTransformer
from src.io.utils import read_config, read_data_csv


def get_classification_labels(
    chat_resampled: pd.DataFrame,
    clip_intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
) -> np.ndarray:
    freq = chat_resampled.index.freq
    window = pd.to_timedelta(freq)

    # Set windows as intervals
    ts = pd.interval_range(
        chat_resampled.index[0], chat_resampled.index[-1] + window, freq=freq
    )

    # Set clip ranges as intervals
    clip_intervals = pd.arrays.IntervalArray(
        [pd.Interval(s, e) for s, e, _ in clip_intervals]
    )

    # Label windows depending on whether window overlaps clip interval
    # 1 - window contains part of clip, 0 - doesn't
    labels = np.zeros(len(ts))

    for clip_interval in clip_intervals:
        idx_overlap = ts.overlaps(clip_interval)
        labels[idx_overlap] = 1

    return labels.astype(int)


def add_lag_lead(
    df: pd.DataFrame, cols: List[str], lag: int, lead: int
) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col + "_lag_" + str(lag)] = df[col].shift(lag)
    for col in cols:
        df[col + "_lead_" + str(lead)] = df[col].shift(-lead)
    return df


def undersample(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    random.seed(random_state)

    minority_count = sum(df["highlight"] == 1)
    df_reset = df.reset_index()
    majority_indices = df_reset[df_reset["highlight"] == 0].index
    majority_undersampled_indices = random.sample(
        list(majority_indices), minority_count
    )
    undersampled_indices = sorted(
        list(majority_undersampled_indices)
        + list(df_reset[df_reset["highlight"] == 1].index)
    )

    return df.iloc[undersampled_indices]


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("mode")
def main(config_path, mode):
    config = read_config(config_path)
    if mode == "train":
        raw_data_path = config["train_data_paths"]["raw_data_path"]
        processed_data_path = config["train_data_paths"]["processed_data_path"]
        resampled_data_path = config["train_data_paths"]["resampled_data_path"]
    elif mode == "test":
        raw_data_path = config["test_data_paths"]["raw_data_path"]
        processed_data_path = config["test_data_paths"]["processed_data_path"]
        resampled_data_path = config["test_data_paths"]["resampled_data_path"]
    else:
        raise ValueError("Invalid mode")

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Build nonoverlapping clips
    clips_df = pd.read_csv(raw_data_path["clips_csv"])
    clips = ClipTransformer(clips_df)
    merged_intervals = clips.select_relevant().merge_overlapping_intervals()

    # Read processed chat data
    chat_df = read_data_csv(processed_data_path["chat_csv"], dates_cols=["time"])
    chat_config = config["processing_config"]
    chat = ChatTransformer(chat_df, chat_config)

    # Resample processed chat data
    chat_resampled = chat.resample()
    logger.info("Resampling chat data")

    # Label target variable
    chat_resampled["highlight"] = get_classification_labels(
        chat_resampled, merged_intervals
    )

    # Add lag 1 and lead 1
    chat_resampled = add_lag_lead(chat_resampled, chat_resampled.columns[:-1], 1, 1)
    chat_resampled = chat_resampled.drop(
        [chat_resampled.index[0], chat_resampled.index[-1]]
    )

    # Balance classes
    if mode == "train":
        random_state = config["undersample_config"]["random_state"]
        chat_resampled = undersample(chat_resampled, random_state)

    # Save resampled chat data
    chat_resampled.to_csv(resampled_data_path["chat_csv"])
    logger.info("Saving resampled chat data")


if __name__ == "__main__":
    main()
