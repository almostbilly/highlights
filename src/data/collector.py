import asyncio
import logging
import os
import sys

import asyncclick as click
from dotenv import find_dotenv, load_dotenv

from src.data.parser import ChatParser, ClipParser
from src.io.utils import read_config


@click.command()
@click.option("--client_id", envvar="CLIENT_ID")
@click.option("--secret_key", envvar="SECRET_KEY")
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("mode")
async def collect_raw_data(client_id, secret_key, config_path, mode):
    config = read_config(config_path)

    if mode == "train":
        video_id = config["data_collector_config"]["video_id_train"]
        raw_data_path = config["train_data_paths"]["raw_data_path"]
    elif mode == "test":
        video_id = config["data_collector_config"]["video_id_test"]
        raw_data_path = config["test_data_paths"]["raw_data_path"]
    else:
        raise ValueError("Invalid mode")

    n_batches = config["data_collector_config"]["n_batches"]

    clipParser = ClipParser(client_id, secret_key)
    clips = clipParser.get_clips(video_id)

    chatParser = ChatParser(client_id, secret_key)
    chat = await chatParser.get_chat(video_id, n_batches)

    logger = logging.getLogger(__name__)
    logger.info("Collecting raw data")

    clips.to_csv(raw_data_path["clips_csv"], index=False)
    chat.to_csv(raw_data_path["chat_csv"], index=False)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    CLIENT_ID = os.getenv("CLIENT_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(collect_raw_data())
