import logging
import os
import random
from datetime import datetime
from typing import List, Tuple

import click
import numpy as np
import pandas as pd
from tsai.all import balance_idx

from src.data.parser.twitch_parser import TwitchParser
from src.features.transformers import ChatTransformer, ClipTransformer
from src.io.utils import read_config, read_data_csv

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def get_classification_labels(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq: str,
    clip_intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
):

    # Set windows as intervals
    ts = pd.interval_range(start=start_time, end=end_time, freq=freq)

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

    return labels.astype(int), ts


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

    if mode not in ["train", "test"]:
        raise ValueError("Invalid mode")

    paths = config[f"{mode}_data_paths"]

    raw_data_path = paths["raw_data_path"]
    processed_data_path = paths["processed_data_path"]
    resampled_data_path = paths["resampled_data_path"]
    labels_path = paths["labels_path"]

    # Build nonoverlapping clips
    clips_df = pd.read_csv(raw_data_path["clips_csv"])
    logger.info("Selecting relevant clips")
    clips = ClipTransformer(clips_df)
    merged_intervals = clips.select_relevant().merge_overlapping_intervals()

    # Read processed chat data
    chat_df = read_data_csv(processed_data_path["chat_csv"], dates_cols=["time"])
    chat_config = config["processing_config"]

    # Set count window size
    count_window = chat_config["count_window"]
    count_window = str(count_window) + "s"

    # Set context window size
    window_size = chat_config["window_size"]
    freq = str(window_size) + "s"

    # Get video duration
    CLIENT_ID = os.getenv("CLIENT_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")
    parser = TwitchParser(CLIENT_ID, SECRET_KEY)
    video_id = config["data_collector_config"]["video_id_train"]
    video = parser.get_video(video_id)
    duration_dt = datetime.strptime(video["duration"], "%Hh%Mm%Ss")

    # Label target variable
    logger.info("Labelling chat data")
    start_time = pd.Timestamp("1970-01-01 00:00:00")
    end_time = start_time + pd.DateOffset(
        hours=duration_dt.hour, minutes=duration_dt.minute, seconds=duration_dt.second
    )
    end_time = end_time.floor(freq)

    labels, time_intervals = get_classification_labels(
        start_time, end_time, freq, merged_intervals
    )

    # Balance classes
    if mode == "train":
        logger.info("Undersampling chat data")
        random_state = config["undersample_config"]["random_state"]

        labels_idx = np.sort(
            balance_idx(labels, strategy="undersample", random_state=random_state)
        )

        labels = labels[labels_idx]
        intervals = time_intervals[labels_idx].values

        message_mask = pd.Series(data=False, index=chat_df.index)

        count_ranges = []
        for interval in intervals:
            s, e = pd.to_datetime(interval.left), pd.to_datetime(interval.right)
            # Select messages that is in undersampled set of context windows
            message_mask = message_mask | (
                (chat_df["time"] >= s) & (chat_df["time"] < e)
            )
            #
            count_ranges.append(
                pd.date_range(s, e, freq=count_window, inclusive="left")
            )

        chat_df = chat_df[message_mask]

        # Interpolate missing count_windows in each context window
        count_window_idx = pd.DatetimeIndex(
            pd.concat([pd.Series(range) for range in count_ranges])
        )
    else:
        # Extrapolate begin and end context windows chat data
        count_window_idx = pd.date_range(
            start_time, end_time, freq=count_window, inclusive="left"
        )

    # Resample processed chat data
    logger.info("Resampling chat data")
    chat = ChatTransformer(chat_df, chat_config)
    chat_resampled = chat.resample()

    # Fill extrapolate/interpolate (depending on undersample) count windows
    chat_resampled = chat_resampled.reindex(count_window_idx, fill_value=0.0)

    # Save resampled chat data
    logger.info("Saving resampled chat data")
    chat_resampled.to_csv(resampled_data_path["chat_csv"], index_label="time")
    pd.DataFrame(labels).to_csv(labels_path, header=None, index=None)


if __name__ == "__main__":
    main()
