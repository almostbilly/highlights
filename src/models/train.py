from typing import List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
from mlflow.models import infer_signature
from src.io.utils import read_data_csv

load_dotenv(find_dotenv())


def construct_intervals(sequence: np.ndarray) -> List[Tuple[int, int]]:
    intervals = []
    start = None
    for i, num in enumerate(sequence):
        if num == 1 and start is None:
            start = i
        elif num == 0 and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(sequence) - 1))
    return intervals


def actualize(sequence: np.ndarray, max_zeros: int) -> np.ndarray:
    new_sequence = sequence.copy()
    intervals = construct_intervals(sequence)
    for x, y in zip(intervals[:-1], intervals[1:]):
        num_zeros = y[0] - x[1] - 1
        if num_zeros <= max_zeros:
            new_sequence[x[1] + 1 : y[0]] = [1] * num_zeros
    return new_sequence


def split_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def eval_metrics(y_test, y_pred):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["precision"] = precision_score(y_test, y_pred)
    metrics["recall"] = recall_score(y_test, y_pred)
    metrics["f1"] = f1_score(y_test, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_test, y_pred)
    return metrics


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def train(config: DictConfig) -> Optional[float]:
    """
    Contains an example training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    chat_train_path = config["train_data_paths"]["resampled_data_path"]["chat_csv"]
    chat_test_path = config["test_data_paths"]["resampled_data_path"]["chat_csv"]
    target = config["target"]

    chat_train = read_data_csv(chat_train_path, dates_cols=["time"])
    chat_train = chat_train.set_index("time")
    chat_test = read_data_csv(chat_test_path, dates_cols=["time"])
    chat_test = chat_test.set_index("time")

    X_train, y_train = split_data(chat_train, target)
    X_test, y_test = split_data(chat_test, target)

    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    remote_registry_uri = config["mlflow_config"]["mlflow_registry_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_registry_uri(remote_registry_uri)

    model_class_components = config.models.model._target_.split(".")
    model_name = model_class_components[-1]
    mlflow.set_experiment(f"{model_name}")

    model = instantiate(config["models"]["model"])

    with mlflow.start_run(nested=True):
        optimized_metric = config["optimized_metric"]
        if optimized_metric not in [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc-auc",
        ]:
            raise ValueError("Metric for hyperparameter optimization not found!")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        max_zeros = 0
        if config["actualize_config"]["enable"]:
            max_zeros = config["actualize_config"]["max_zeros"]
            y_pred = actualize(y_pred, max_zeros)

        mlflow.log_params(model.get_params())
        mlflow.log_param("max_zeros", max_zeros)

        metrics = eval_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("roc-auc", metrics["roc_auc"])

        signature = infer_signature(X_train, y_pred)

        flavor = model_class_components[0]
        getattr(mlflow, flavor).log_model(model, "model", signature=signature)

        score = metrics[optimized_metric]

    return score


if __name__ == "__main__":
    train()
