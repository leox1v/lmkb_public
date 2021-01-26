from typing import List
import pandas as pd
import numpy as np

def precisions_per_relation(df: pd.DataFrame, ks: List[int]):
    # Compute Precisions.
    def compute_precision(target: str, predictions: List[str], k: int=1) -> float:
        return target in predictions[:k]

    precisions = {k: df.apply(lambda row: compute_precision(
        target=row.y.lower(),
        predictions=[y.lower() for y in row.y_pred],
        k=k).mean(), axis=1)
        for k in ks}
    return precisions
