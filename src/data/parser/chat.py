import itertools
import os
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from src.data.parser.thirdparties import ThirtPartyEmotesParser
from src.data.parser.twitch_parser import TwitchParser


class ChatParser(TwitchParser):
    def __init__(self, client_id: str, secret_key: str):
        super().__init__(client_id, secret_key)
        self.all_emotes = None

    def get_comment_data(self, comment):
        created_at = pd.to_datetime(comment["contentOffsetSeconds"], unit="s")
        author_name = (
            comment["commenter"]["displayName"] if comment["commenter"] else None
        )

        text = []
        emotes = []
        message_fragments = []
        for fragment in comment["message"]["fragments"]:
            message_fragments.append(fragment["text"])
            for word in fragment["text"].split():
                if word in self.all_emotes:
                    emotes.append(word)
                else:
                    text.append(word)
        message = "".join(message_fragments)
        text = " ".join(text)
        # unique_emotes = set(emotes)

        # return [created_at, author_name, message, text, emotes, unique_emotes]
        return [created_at, author_name, message, text, emotes]

    async def get_page_comments(
        self, video_id: str, video_start: int = None, cursor: str = None, session=None
    ) -> List[Dict]:
        """Getting chat data of the specified VOD
        Args:
            video_id (str): Twitch VOD ID whose clips are requested
        Returns:
            list[dict]: response in JSON format
        """
        # the base url for making api requests
        url = "https://gql.twitch.tv/gql"
        # authorization information
        headers = {
            "Client-Id": os.getenv("TWITCH_CLIENT_ID"),
            "Content-Type": "application/json",
        }
        # GraphQL query
        payload = {
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": {"videoID": video_id},
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a",
                }
            },
        }
        if video_start and not cursor:
            payload["variables"]["contentOffsetSeconds"] = video_start
        elif cursor and not video_start:
            payload["variables"]["cursor"] = cursor

        async with session.post(url, headers=headers, json=payload) as response:
            response = await response.json()
        return response

    async def get_video_comments(self, video_id, video_start, video_end, session):
        cursors = []
        comments = []
        response = await self.get_page_comments(video_id, video_start, session=session)

        comments_data = response["data"]["video"]["comments"]["edges"]
        for comment in comments_data:
            if (
                comment["node"]["contentOffsetSeconds"] > video_start
                and comment["node"]["contentOffsetSeconds"] < video_end
            ):
                comment = self.get_comment_data(comment["node"])
                comments.append(comment)
        if response["data"]["video"]["comments"]["pageInfo"]["hasNextPage"]:
            cursor = response["data"]["video"]["comments"]["edges"][-1]["cursor"]
            cursors.append(cursor)

        while cursor:
            response = await self.get_page_comments(
                video_id, video_start=None, cursor=cursor, session=session
            )

            comments_data = response["data"]["video"]["comments"]["edges"]
            for comment in comments_data:
                if (
                    comment["node"]["contentOffsetSeconds"] > video_start
                    and comment["node"]["contentOffsetSeconds"] < video_end
                ):
                    comment = self.get_comment_data(comment["node"])
                    comments.append(comment)
            if comments_data[-1]["node"]["contentOffsetSeconds"] >= video_end:
                break
            if response["data"]["video"]["comments"]["pageInfo"]["hasNextPage"]:
                cursor = response["data"]["video"]["comments"]["edges"][-1]["cursor"]
                cursors.append(cursor)
            else:
                cursor = None

        return comments

    async def get_chat(self, video_id, n_batches) -> pd.DataFrame:
        video = self.get_video(video_id)
        broadcaster_id = video["user_id"]
        duration_dt = datetime.strptime(video["duration"], "%Hh%Mm%Ss")
        video_duration = timedelta(
            hours=duration_dt.hour,
            minutes=duration_dt.minute,
            seconds=duration_dt.second,
        ).seconds
        batch_size = video_duration / n_batches
        video_starts = [int(batch_size * i) for i in range(n_batches)]
        video_ends = [video_start + batch_size for video_start in video_starts]

        emotesParser = ThirtPartyEmotesParser()
        self.all_emotes = emotesParser.get_all_channel_emotes(broadcaster_id)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(n_batches):
                tasks.append(
                    self.get_video_comments(
                        video_id, video_starts[i], video_ends[i], session
                    )
                )
            results = await tqdm_asyncio.gather(*tasks)

        chat = list(itertools.chain.from_iterable(results))
        chat_df = pd.DataFrame(
            chat, columns=["time", "author", "message", "text", "emotes"]
        )
        return chat_df
