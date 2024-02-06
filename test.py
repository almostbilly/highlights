import asyncio
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv

from src.data.collector import collect_raw_data

load_dotenv(find_dotenv())
CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
VIDEO_ID_TRAIN = os.getenv("VIDEO_ID_TRAIN")

# def get_clips_data():
#     clipParser = ClipParser(CLIENT_ID, SECRET_KEY)
#     clips = clipParser.get_clips(video_id=VIDEO_ID_TRAIN)
#     # remove unwanted clips
#     # clips = clips.drop(clips[(clips['duration'] == 0) | (clips['view_count'] == 1)].index)
#     print(clips.head(2))

# async def get_chat_data():
#     chatParser = ChatParser(CLIENT_ID, SECRET_KEY)
#     n_batches = 32
#     chat = await chatParser.get_chat(VIDEO_ID_TRAIN, n_batches)
#     print(chat.head())

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(collect_raw_data(CLIENT_ID, SECRET_KEY, VIDEO_ID_TRAIN, 32))
    # get_clips_data()
    # if sys.platform == 'win32':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # asyncio.run(get_chat_data())
