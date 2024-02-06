import hydra
from omegaconf import DictConfig

import mlflow
from mlflow import MlflowClient


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def register_best_model(config: DictConfig):
    model_name = config["models"]["model"]["_target_"].split(".")[-1]
    optimized_metric = config["optimized_metric"]

    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(
        experiment_names=[model_name],
        order_by=[f"metrics.{optimized_metric} DESC"],
    )
    best_run = runs.iloc[0]

    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name, await_registration_for=0)

    # Register model name in the model registry
    client = MlflowClient()
    print(dict(client.search_model_versions(f"name='{model_name}'")[0])["version"])
    # client.create_registered_model(model_name)

    # # Create a new version of the rfr model under the registered model name
    # run_id = best_run["run_id"]
    # model_uri = f"runs:/{run_id}/model"
    # model_src = RunsArtifactRepository.get_underlying_uri(model_uri)
    # client.create_model_version(model_name, model_src, run_id)


if __name__ == "__main__":
    register_best_model()
