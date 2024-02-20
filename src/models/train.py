from typing import List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from hydra import compose, initialize
from hydra.utils import instantiate
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tsai.all import TSUnwindowedDataset

from src.io.utils import read_data_csv

load_dotenv(find_dotenv())


class ScalerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, input_data):
        return self.scaler.transform(input_data)

    def predict(self, context, model_input):
        pass


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


def load_data(config: DictConfig, mode: str = "train"):
    data_path = config[f"{mode}_data_paths"]["resampled_data_path"]["chat_csv"]
    labels_path = config[f"{mode}_data_paths"]["labels_path"]
    X = read_data_csv(data_path, dates_cols=["time"]).set_index("time")
    y = pd.read_csv(labels_path, header=None)
    return X, y


def apply_sliding_window(X, y, window_size, stride):
    X = TSUnwindowedDataset(X.to_numpy(), window_size=window_size, stride=stride)[:][
        0
    ].data
    y = np.asarray(y).squeeze(1)
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
    Training pipeline for hyperparameters tuning.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    window_size = config["processing_config"]["window_size"]

    X_train_df, y_train_df = load_data(config, mode="train")
    X_test_df, y_test_df = load_data(config, mode="test")

    # SlidingWindow(window_size=seq_len): (full_seq_len, n_features) -> (n_seq, n_features, seq_len)
    X_train, y_train = apply_sliding_window(
        X_train_df, y_train_df, window_size=window_size, stride=window_size
    )
    X_test, y_test = apply_sliding_window(
        X_test_df, y_test_df, window_size=window_size, stride=window_size
    )

    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    remote_registry_uri = config["mlflow_config"]["mlflow_registry_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_registry_uri(remote_registry_uri)

    model_class_components = config.model._target_.split(".")
    model_name = model_class_components[-1]
    mlflow.set_experiment(f"{model_name}")

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

        scaler = instantiate(config["scaler"])
        X_train = scaler.fit_transform(X_train)

        model = instantiate(config["model"])

        model.fit(X_train, y_train)

        X_test = scaler.transform(X_test)
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

        scaler_wrapper = ScalerWrapper(scaler)
        mlflow.pyfunc.log_model(python_model=scaler_wrapper, artifact_path="scaler")

        signature = infer_signature(X_train, y_pred)

        flavor = model_class_components[0]
        getattr(mlflow, flavor).log_model(model, "model", signature=signature)

        score = metrics[optimized_metric]

    return score


def main():
    # Hydra Multirun hyperparameter tuning
    train()

    with initialize(version_base=None, config_path="../conf"):
        config = compose(config_name="train")
    model_class_components = config.model._target_.split(".")
    model_name = model_class_components[-1]
    experiment = mlflow.get_experiment_by_name(model_name)

    config = OmegaConf.load("src/conf/params.yaml")
    OmegaConf.update(config, "experiment_id", experiment.experiment_id)
    with open("src/conf/params.yaml", "w") as f:
        OmegaConf.save(config, f)


if __name__ == "__main__":
    main()
