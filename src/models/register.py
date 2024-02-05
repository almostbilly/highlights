import hydra
from omegaconf import DictConfig

import mlflow


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def register_best_model(config: DictConfig):
    model_name = config["models"]["model"]["_target_"].split(".")[-1]
    optimized_metric = config["optimized_metric"]

    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    remote_registry_uri = config["mlflow_config"]["mlflow_registry_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_registry_uri(remote_registry_uri)

    runs = mlflow.search_runs(
        experiment_names=[model_name],
        order_by=[f"metrics.{optimized_metric} DESC"],
    )
    best_run = runs.iloc[0]

    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name, await_registration_for=0)


if __name__ == "__main__":
    register_best_model()
