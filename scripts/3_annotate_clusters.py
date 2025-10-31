"""Extract key talking points from ad clusters."""

import logging

import hydra
import joblib
import pandas as pd
from hbdscan import HBDSCAN
from jinja2 import Environment
from omegaconf import DictConfig
from tqdm import tqdm
from utils import config_to_dict, get_cluster
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

logger = logging.getLogger(__name__)


def classification_pipeline(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts_cfg: DictConfig,
    ads: list,
    topics: list[str],
    moral_foundations: list[tuple[str, str]],
    skip_summary: bool = False,
) -> tuple[str, str, str, str]:
    """
    Given a set of ads, classify them based on the topics and moral foundations.

    Parameters:
        lm (guidance): LLM guidance object with the context
        sampling_params (SamplingParams): Sampling parameters for the LLM
        prompt_cfg (DictConfig): Configuration for the prompts
        ads (list): List of ads to classify
        topics (list): List of topics to classify the ads
        moral_foundations (list[tuple[str, str]]): List of moral foundations (name, description)
        skip_summary (bool): Whether to skip the summary generation

    Returns:
        str: The topic of the ads
        str: The stance of the ads (in favor of or against)
        str: The moral foundation of the ads
        str: The summary of the ads and the classification
    """

    jinja_env = Environment()

    system_prompt = prompts_cfg.system + prompts_cfg.ad_separator + prompts_cfg.add_separator.join(ads)
    prompt = [{"role": "system", "content": system_prompt}]

    # Topic selection prompt
    user_msg = prompts_cfg.topic + ", ".join(topics)
    user_prompt = [{"role": "user", "content": user_msg}]

    # Topic classification
    topic_selection = GuidedDecodingParams(choices=topics)
    sampling_params.guided_decoding = topic_selection

    # Generate the topic
    outputs = llm.generate(prompt + user_prompt, sampling_params=sampling_params)
    topic = outputs[0].completions[0].text.strip()

    # Stance classification
    template = jinja_env.from_string(prompts_cfg.leaning)
    stance = template.render(topic=topic)
    user_prompt = [{"role": "user", "content": stance}]

    # Set up guided decoding for stance classification
    stance_options = ["in favor of", "against"]
    stance_decoding = GuidedDecodingParams(choices=stance_options)
    sampling_params.guided_decoding = stance_decoding

    # Generate the stance
    outputs = llm.generate(prompt + user_prompt, sampling_params=sampling_params)
    stance = outputs[0].completions[0].text.strip()

    # Moral foundation classification
    template = jinja_env.from_string(prompts_cfg.moral_foundation)
    moral_foundation = template.render(topic=topic, stance=stance)
    moral_foundation += " - " + "\n - ".join([f'"{name}: {desc}"' for name, desc in moral_foundations])

    # Set up guided decoding for moral foundation classification
    moral_options = [name for name, _ in moral_foundations]
    moral_decoding = GuidedDecodingParams(choices=moral_options)
    sampling_params.guided_decoding = moral_decoding

    # Generate the moral foundation
    user_prompt = [{"role": "user", "content": moral_foundation}]
    outputs = llm.generate(prompt + user_prompt, sampling_params=sampling_params)
    morals = outputs[0].completions[0].text.strip()

    # Summary generation
    summary = ""
    if not skip_summary:
        template = jinja_env.from_string(prompts_cfg.summarize)
        summary_prompt = template.render(topic=topic, stance=stance, moral_foundation=morals)
        user_prompt = [{"role": "user", "content": summary_prompt}]
        sampling_params.guided_decoding = None
        sampling_params.stop = ["."]
        outputs = llm.generate(prompt + user_prompt, sampling_params=sampling_params)
        summary = outputs[0].completions[0].text.strip()

        # Final summary formatting
        template = jinja_env.from_string(prompts_cfg.summary)
        summary = template.render(topic=topic, stance=stance, moral_foundation=morals, summary=summary)

    return topic, stance, morals, summary


def try_classify(
    df: pd.DataFrame,
    cluster: int,
    llm: LLM,
    sampling_params: SamplingParams,
    prompts_cfg: DictConfig,
    clusterer: HBDSCAN,
    topics: list[str],
    moral_foundations: list[tuple[str, str]],
    **get_cluster_kwargs,
) -> tuple[str, str, str, str] | tuple[None, None, None, None]:
    try:
        ads = get_cluster(df, cluster, clusterer, **get_cluster_kwargs)
        return classification_pipeline(llm, sampling_params, prompts_cfg, ads, topics, moral_foundations)
    except Exception as e:
        logger.error(f"Error getting ads for cluster {cluster}: {e}")
        return None, None, None, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize the LLM and sampling parameters
    llm = LLM(cfg.llm)
    params = config_to_dict(cfg, "sampling_params")
    sampling_params = SamplingParams(**params)
    logger.info(f"Using LLM: {cfg.llm} with sampling params: {sampling_params}")

    # Load the moral foundations
    moral_foundations = list(cfg.moral_foundations.items())

    # Load the data
    df = pd.read_csv(cfg.data.path)
    logger.info(f"Loaded {len(df)} ads from {cfg.data.path}")

    # Load the clusterer and cluster parameters
    clusterer = joblib.load(cfg.data.clusterer_path)
    n_clusters = int(max(clusterer.labels_)) + 1
    cluster_params = config_to_dict(cfg, "cluster_filter_params")
    logger.info(f"Loaded clusterer from {cfg.data.clusterer_path} with {n_clusters} clusters")

    # Load the seed set of topics
    topics = getattr(cfg, "seed_set", [])

    # Populate topic set
    results = []
    for cluster in tqdm(range(n_clusters), desc="Processing clusters"):
        topic, stance, morals, summary = try_classify(
            df,
            cluster,
            llm,
            sampling_params,
            cfg.prompts.classification_pipeline,
            clusterer,
            topics,
            moral_foundations,
            **cluster_params,
        )
        results.append((topic, stance, morals, summary))

    # Save the results
    results_df = pd.DataFrame(results, columns=["topic", "stance", "morals", "summary"])
    results_df.to_csv(cfg.data.annotation_path, index=False)

    # Final logging
    logger.info(f"Saved annotations to {cfg.data.annotation_path}")
    logger.info(f"Final annotation results:\n{results_df}")


if __name__ == "__main__":
    main()  # type: ignore
