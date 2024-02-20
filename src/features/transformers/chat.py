import itertools
import re
from typing import Dict

import numpy as np
import pandas as pd


class ChatTransformer:
    def __init__(self, chat_df: pd.DataFrame, config: Dict) -> None:
        self.df = chat_df
        self.config = config

    def build_features(self) -> "ChatTransformer":
        df = self.df
        # initialize params
        min_words = self.config["shortness_threshold"]
        # replace NaN values with ""
        df["text"] = df["text"].fillna("")
        # extract unique emotes per message
        df["unique_emotes"] = df["emotes"].apply(np.unique)
        # define whether a message is a copypaste
        is_short = df["text"].str.split().str.len() < min_words
        is_duplicated = df["text"].str.lower().duplicated(keep=False)
        df["paste"] = (is_duplicated) & (~is_short)
        # count number of mentions in a message
        df["mention"] = df["text"].apply(lambda x: len(re.findall(r"@\b\w+", x)))
        return self

    def resample(self):
        df = self.df
        df = df.set_index("time")
        min_words = self.config["shortness_threshold"]
        count_window = self.config["count_window"]
        freq = str(count_window) + "s"
        f = df.index.floor(freq)
        df_grouped = df.groupby(f)
        df_resampled = pd.concat(
            [
                df_grouped["text"].agg(
                    lambda x: sum(x.str.split().str.len() < min_words)
                ),
                df_grouped["unique_emotes"].agg(
                    lambda x: len(list(itertools.chain.from_iterable(x)))
                ),
                df_grouped["emotes"].count(),
                df_grouped["author"].nunique(),
                df_grouped["paste"].sum(),
                df_grouped["mention"].sum(),
            ],
            axis=1,
        )
        df_resampled.columns = [
            "cnt_short",
            "nunique_emotes",
            "cnt_messages",
            "nunique_authors",
            "cnt_paste",
            "cnt_mentions",
        ]
        return df_resampled
