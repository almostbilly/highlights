import ast
from typing import Dict, List, Union

import pandas as pd
import yaml


def read_data_csv(
    data_path: str, dates_cols: Union[List[str], bool] = False
) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=dates_cols)

    array_columns = [
        col
        for col, dtype in df.dtypes.items()
        if dtype == "object"
        and df[col]
        .apply(lambda x: isinstance(x, str) and x.startswith("[") and x.endswith("]"))
        .all()
    ]

    for col in array_columns:
        df[col] = df[col].apply(ast.literal_eval)

    return df


def read_config(config_path: str) -> Dict:
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def write_config(config_path: str, config: Dict):
    with open(config_path, "w") as yaml_file:
        yaml.dump(config, yaml_file)
