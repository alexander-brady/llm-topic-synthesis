"""Extract key talking points from ad clusters."""

import logging

import hydra
import joblib
import pandas as pd
from jinja2 import Environment
from omegaconf import DictConfig
from tqdm import tqdm
from utils import config_to_dict, get_cluster
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

logger = logging.getLogger(__name__)


def extend_annotation_list(
    llm: LLM, sampling_params: SamplingParams, prompts_cfg: DictConfig, ads: list, topics: list[str]
) -> str | None:
    """
    Given a set of ads, find a topic that describes the main issue, if none of the others are relevant.

    Parameters:
        llm (LLM): LLM object to use for classification
        sampling_params (SamplingParams): Sampling parameters for the LLM
        prompts_cfg (DictConfig): Configuration for the prompts to use
        ads (list): List of ads to classify
        topics (list): List of possible topics to classify the ads

    Returns:
        str: A new topic that describes the ads, or none if there is a relevant topic in the list
    """
    jinja_env = Environment()

    system_prompt = prompts_cfg.system + prompts_cfg.ad_separator + prompts_cfg.add_separator.join(ads)
    prompt = [{"role": "system", "content": system_prompt}]

    # Create the user prompt
    template = jinja_env.from_string(prompts_cfg.binary)
    options = ", ".join(topics)

    prompt.append({"role": "user", "content": template.render(options=options)})

    # Set up guided decoding to force "yes" or "no" answers
    binary_decoding = GuidedDecodingParams(choices=["yes", "no"])
    sampling_params.guided_decoding = binary_decoding

    # Generate the response
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    response = outputs[0].completions[0].text.strip().lower()

    # If no new topic is found, return None
    if response == "yes":
        return None

    # Otherwise, synthesize a new topic
    prompt.extend(
        [
            {"role": "assistant", "content": prompts_cfg.none_relevant},
            {"role": "user", "content": prompts_cfg.new_topic[0]},
            {"role": "assistant", "content": prompts_cfg.new_topic[1]},
        ]
    )

    # Reset guided decoding for topic synthesis
    sampling_params.guided_decoding = None
    sampling_params.stop = ['"']

    # Generate the new topic
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    new_topic = outputs[0].completions[0].text.strip()

    return new_topic


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize the LLM and sampling parameters
    llm = LLM(cfg.llm)
    params = config_to_dict(cfg, "sampling_params")
    sampling_params = SamplingParams(**params)
    logger.info(f"Using LLM: {cfg.llm} with sampling params: {sampling_params}")

    # Load the data
    df = pd.read_csv(cfg.data.path)
    logger.info(f"Loaded {len(df)} ads from {cfg.data.path}")

    # Load the clusterer and cluster parameters
    clusterer = joblib.load(cfg.data.clusterer_path)
    n_clusters = int(max(clusterer.labels_)) + 1
    cluster_params = config_to_dict(cfg, "cluster_filter_params")
    logger.info(f"Loaded clusterer from {cfg.data.clusterer_path} with {n_clusters} clusters")

    # Populate the topics
    topics = getattr(cfg, "seed_set", [])
    for cluster in tqdm(range(n_clusters), desc="Processing clusters"):
        ads = get_cluster(df, cluster, clusterer, **cluster_params)
        topic = extend_annotation_list(llm, sampling_params.clone(), cfg.prompts.topic_synthesis, ads, topics)
        if topic:
            logger.info(f"[Cluster {cluster}] New topic discovered: {topic}")
        topics.append(topic)

    # Save the final topic list
    logger.info(f"Final topic list: {topics}")
    with open(cfg.data.topics_path, "w") as f:
        f.write("\n".join(topics))

    # Final log
    logger.info(f"Saved topics to {cfg.data.topics_path}")
    logger.info("Final topic list:\n" + " ,".join(topics))


if __name__ == "__main__":
    main()  # type: ignore
