import logging

import click

from src.features.transformers import ChatTransformer
from src.io.utils import read_config, read_data_csv


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("mode")
def main(config_path, mode):
    config = read_config(config_path)
    if mode == "train":
        raw_data_path = config["train_data_paths"]["raw_data_path"]
        processed_data_path = config["train_data_paths"]["processed_data_path"]
    elif mode == "test":
        raw_data_path = config["test_data_paths"]["raw_data_path"]
        processed_data_path = config["test_data_paths"]["processed_data_path"]
    else:
        raise ValueError("Invalid mode")

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Build novel chat features
    chat_df = read_data_csv(raw_data_path["chat_csv"], dates_cols=["time"])
    chat_config = config["processing_config"]
    chat = ChatTransformer(chat_df, chat_config)
    chat = chat.build_features()
    logger.info("Processing chat data")

    # Save processed chat data
    chat.df.to_csv(processed_data_path["chat_csv"], index=False)
    logger.info("Saving processed chat data")


if __name__ == "__main__":
    main()
