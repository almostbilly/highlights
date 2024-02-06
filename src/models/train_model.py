import time
from collections import namedtuple
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import all_estimators
from xgboost import XGBClassifier

from src.io.utils import read_config, read_data_csv


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


def initialize_models(config: Dict):
    all_classifiers = {
        name: obj for name, obj in all_estimators() if issubclass(obj, ClassifierMixin)
    }
    all_classifiers["XGBClassifier"] = XGBClassifier

    # Classifiers = namedtuple("Classifiers", __all_classifiers.keys())(
    #     *__all_classifiers.values()
    # )
    # print(Classifiers)

    models = config["models"]

    # classifier = all_classifiers[models[0]["name"]]
    # print(classifier)

    initialized_models = []

    for model in models:
        clf_name = model["name"]
        clf_params = model["params"] if "params" in model else None

        # If the classifier is not in the dictionary, exit with error
        if clf_name not in all_classifiers.keys():
            raise ValueError(f"Classifier {clf_name} not found.")

        classifier = all_classifiers[clf_name]

        # classifier = getattr(Classifiers, clf_name)

        # Verify if all parameters passed by the yaml are valid
        # classifier parameters
        if clf_params:
            attributes = dir(classifier())
            invalid_arg = [arg for arg in clf_params if arg not in attributes]

            # If there are invalid parameters, print an error message and exit
            if invalid_arg:
                raise ValueError(f"Error: Invalid parameters {invalid_arg}")

        # Initialize the classifier with the specified parameters
        clf_instance = classifier(**clf_params) if clf_params else classifier()
        initialized_models.append(clf_instance)

    return initialized_models


def split_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def eval_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path):
    config = read_config(config_path)
    chat_train_path = config["train_data_paths"]["resampled_data_path"]["chat_csv"]
    chat_test_path = config["test_data_paths"]["resampled_data_path"]["chat_csv"]
    target = config["target"]

    classifiers = initialize_models(config)

    chat_train = read_data_csv(chat_train_path, dates_cols=["time"])
    chat_train = chat_train.set_index("time")
    chat_test = read_data_csv(chat_test_path, dates_cols=["time"])
    chat_test = chat_test.set_index("time")

    X_train, y_train = split_data(chat_train, target)
    X_test, y_test = split_data(chat_test, target)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # experiment = mlflow.set_experiment(mlflow_config["experiment_name"])
    # experiment_id = experiment.experiment_id
    mlflow.set_tracking_uri(remote_server_uri)

    for clf in classifiers:
        mlflow.set_experiment(f"Model {type(clf).__name__}")
        with mlflow.start_run():
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            max_zeros = 0
            if config["actualize_config"]["enable"]:
                max_zeros = config["actualize_config"]["max_zeros"]
                y_pred = actualize(y_pred, max_zeros)

            for param, value in clf.get_params().items():
                mlflow.log_param(f"{type(clf).__name__}_{param}", value)
            mlflow.log_param("max_zeros", max_zeros)

            accuracy, precision, recall, f1, roc_auc = eval_metrics(y_test, y_pred)

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("Roc-auc", roc_auc)

            signature = infer_signature(X_train, y_pred)

            if isinstance(clf, XGBClassifier):
                mlflow.xgboost.log_model(clf, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(clf, "model", signature=signature)


if __name__ == "__main__":
    train()
