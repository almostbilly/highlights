import logging
from typing import List, Tuple

import pandas as pd


class ClipTransformer:
    def __init__(self, clips_df: pd.DataFrame) -> None:
        self.df = clips_df

    def select_relevant(self) -> "ClipTransformer":
        df = self.df
        self.df = df.drop(df[(df["duration"] == 0) | (df["view_count"] == 1)].index)
        return self

    def make_intervals(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        started_at = self.df["vod_offset"]
        ended_at = started_at + self.df["duration"]
        view_count = self.df["view_count"]

        started_at = pd.to_datetime(started_at, unit="s")
        ended_at = pd.to_datetime(ended_at, unit="s")

        clip_intervals = []
        for start, end, views in zip(started_at, ended_at, view_count):
            clip_intervals.append((start, end, views))
        return clip_intervals

    def merge_overlapping_intervals(
        self,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        # Contstuct the ranges
        ranges = self.make_intervals()
        # Sort the ranges based on the start timestamps
        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        combined_ranges = []
        for start, end, view_count in sorted_ranges:
            if not combined_ranges or start > combined_ranges[-1][1]:
                # No overlap with the previous range, add it to the combined list
                combined_ranges.append((start, end, view_count))
            else:
                # There is an overlap, update the end timestamp and sum the view count of the previous range
                combined_ranges[-1] = (
                    combined_ranges[-1][0],
                    max(end, combined_ranges[-1][1]),
                    combined_ranges[-1][2] + view_count,
                )

        # Sort the combined ranges back to the original order
        sorted_combined_ranges = sorted(
            combined_ranges, key=lambda x: x[2], reverse=True
        )

        logger = logging.getLogger(__name__)
        logger.info(
            f"{len(ranges)} clips were received at the input and {len(combined_ranges)} clips were left after merging"
        )

        return sorted_combined_ranges
