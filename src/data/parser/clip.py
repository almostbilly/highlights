import itertools
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from src.data.parser.twitch_parser import TwitchParser


class ClipParser(TwitchParser):
    def __init__(self, client_id: str, secret_key: str):
        super().__init__(client_id, secret_key)

    def get_clips(self, video_id: str = None) -> pd.DataFrame:
        """Getting all clips of the specified VOD
        Args:
            video_id (str): Twitch VOD ID whose clips are requested
        Returns:
            list[dict]: list of clips created during VOD,
                        in descending order of number of views
        """
        if video_id is None:
            raise TypeError("'None' value provided for video_id")
        # GET request to Twitch API /videos
        video = self.get_video(video_id)
        broadcaster_id = video["user_id"]
        duration = video["duration"]
        # Getting start time of stream
        started_at = video["created_at"]
        # Finding end time of stream
        started_at_dt = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ")
        duration_dt = datetime.strptime(duration, "%Hh%Mm%Ss")
        ended_at_dt = started_at_dt + timedelta(
            hours=duration_dt.hour,
            minutes=duration_dt.minute,
            seconds=duration_dt.second,
        )
        ended_at = ended_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # GET request for all clips for each existed page
        has_pages = True
        pages = []
        cursor = ""
        with tqdm() as pbar:
            while has_pages:
                response = self.request_get(
                    "clips",
                    {
                        "broadcaster_id": broadcaster_id,
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "first": 100,
                        "after": cursor,
                    },
                )
                has_pages = bool(response["pagination"])
                if has_pages:
                    cursor = response["pagination"]["cursor"]
                pages.append(response["data"])
                pbar.update(1)
        clips = list(itertools.chain.from_iterable(pages))
        clips_df = pd.DataFrame(clips)
        return clips_df
