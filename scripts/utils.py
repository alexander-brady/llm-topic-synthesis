from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from omegaconf import DictConfig, OmegaConf


def config_to_dict(cfg: DictConfig, field: str) -> dict[str, Any]:
    """Convert a specific field in a DictConfig to a dictionary.

    Args:
        cfg (DictConfig): The configuration object.
        field (str): The field to convert.

    Returns:
        dict[str, any]: The converted dictionary.
    """
    if field not in cfg:
        return {}

    return OmegaConf.to_container(cfg[field], resolve=True)


def get_cluster(
    data: pd.DataFrame,
    cluster: int,
    clusterer: HDBSCAN,
    top_k: int = -1,
    min_prob: float = 0,
    column: str | list[str] | None = "ad_creative_bodies",
    keep_index: bool = False,
) -> list:
    """
    Given a cluster, return the ads in that cluster.

    Parameters:
        data (pd.DataFrame): The data containing the ads
        clusterer (HDBSCAN): The clusterer object, with the labels and probabilities
        cluster (int): The cluster number
        top_k (int): The number of ads to return. If -1, return all ads in the cluster
        min_prob (float): The minimum membership probability to consider an ad in the cluster
        columns (Optional[str|List[str]]): The column to use for the ads. Default is 'ad_creative_bodies'
        keep_index (bool): Whether to keep the index of the ads in the original data

    Returns:
        list: The ads in the cluster, sorted by relevance
    """
    if not column:
        column = data.columns.tolist()

    ads = [
        (data.index[i], data.iloc[i][column] if isinstance(column, str) else tuple(data.iloc[i][col] for col in column))
        for i in np.argsort(clusterer.probabilities_)[::-1]
        if clusterer.labels_[i] == cluster and clusterer.probabilities_[i] >= min_prob
    ]

    if not keep_index:
        ads = [ad[1] for ad in ads]

    return ads[:top_k] if top_k != -1 else ads
