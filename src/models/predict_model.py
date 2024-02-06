import time

import numpy as np
import pandas as pd

from src.models.train_model import construct_intervals


def make_highlight_intervals(sequence: np.ndarray, window_size: int):
    output_df = pd.DataFrame(columns=["start", "end"])
    highlight_intervals = construct_intervals(sequence)
    for s, e in highlight_intervals:
        start = time.strftime("%H:%M:%S", time.gmtime(s * window_size))
        end = time.strftime("%H:%M:%S", time.gmtime((e + 1) * window_size))
        output_df = output_df.append({"start": start, "end": end}, ignore_index=True)
    return output_df
