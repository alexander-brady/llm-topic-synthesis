"""Embed text data to vector representations using pre-trained models and cluster ads based on representation."""

import hydra
import joblib
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def identify_duplicates(embeddings: np.ndarray, threshold: float = 0.95) -> list[int]:
    """Identify duplicate embeddings based on cosine similarity threshold.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        threshold (float): Cosine similarity threshold to consider as duplicates.

    Returns:
        list[int]: indices of embeddings which have a duplicate.
    """
    similarities = cosine_similarity(embeddings)
    to_remove = set()

    x, y = np.triu_indices_from(similarities, k=1)
    for i, j in zip(x, y, strict=False):
        if similarities[i, j] >= threshold:
            to_remove.add(j)

    return list(to_remove)


@hydra.main(version_base="1.3", config_path="../configs", config_name="embedding")
def main(cfg: DictConfig):
    # Load data and model
    df = pd.read_csv(cfg.data.path)
    model = SentenceTransformer(cfg.model)

    # Generate embeddings
    embeddings = model.encode(df[cfg.data.col.content].tolist(), convert_to_numpy=True, show_progress_bar=True)

    # Remove duplicate embeddings
    duplicate_indices = identify_duplicates(embeddings, threshold=cfg.duplicate_threshold)
    df_cleaned = df.drop(index=duplicate_indices).reset_index(drop=True)
    embeddings_cleaned = np.delete(embeddings, duplicate_indices, axis=0)

    # Cluster embeddings
    cluster_params = OmegaConf.to_container(cfg.hdbscan_params, resolve=True)
    clusterer = HDBSCAN(**cluster_params)  # type: ignore
    clusterer.fit(embeddings_cleaned)

    # Add cluster labels to dataframe
    df_cleaned["cluster"] = clusterer.labels_

    # Save cleaned data, embeddings, and clusterer
    df_cleaned.to_csv(cfg.data.path, index=False)
    np.save(cfg.data.embeddings_path, embeddings_cleaned)
    joblib.dump(clusterer, cfg.data.clusterer_path)


if __name__ == "__main__":
    main()  # type: ignore
