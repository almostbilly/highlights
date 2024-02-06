import logging
import os
import time

import mlflow
import numpy as np
import pandas as pd
from hydra import compose, initialize

from src.data.parser.chat import ChatParser
from src.features.resample import add_lag_lead
from src.features.transformers.chat import ChatTransformer
from src.models.train import construct_intervals

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


def make_highlight_intervals(sequence: np.ndarray, window_size: int) -> pd.DataFrame:
    highlight_intervals = construct_intervals(sequence)
    intervals = []

    for s, e in highlight_intervals:
        start = format_time(s * window_size)
        end = format_time((e + 1) * window_size)
        intervals.append({"start": start, "end": end})

    return intervals


def format_time(seconds: int) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


async def predict(video_id: int):
    logger = logging.getLogger("PREDICT")

    logger.info("Reading config")
    with initialize(version_base=None, config_path="../conf"):
        config = compose(config_name="train")

    n_batches = config["data_collector_config"]["n_batches"]

    logger.info("Collecting chat data")
    CLIENT_ID = os.getenv("CLIENT_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")
    chatParser = ChatParser(CLIENT_ID, SECRET_KEY)
    chat = await chatParser.get_chat(video_id, n_batches)

    logger.info("Processing chat data")
    chat_config = config["processing_config"]
    chat = ChatTransformer(chat, chat_config)
    chat = chat.build_features()

    logger.info("Resampling chat data")
    chat_resampled = chat.resample()

    # scaler = load_scaler

    logger.info("Adding lag and lead")
    chat_resampled = add_lag_lead(chat_resampled, chat_resampled.columns, 1, 1)
    chat_resampled = chat_resampled.drop(
        [chat_resampled.index[0], chat_resampled.index[-1]]
    )

    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    remote_registry_uri = config["mlflow_config"]["mlflow_registry_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_registry_uri(remote_registry_uri)

    client = mlflow.MlflowClient()
    model_name = config["models"]["model"]["_target_"].split(".")[-1]
    try:
        last_registered_model = client.search_model_versions(f"name='{model_name}'")[0]
    except Exception:
        logger.exception("No available model")
    model_version = dict(last_registered_model)["version"]
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    prediction = model.predict(chat_resampled)

    highlights_df = make_highlight_intervals(
        prediction, config["processing_config"]["window_size"]
    )

    return highlights_df
