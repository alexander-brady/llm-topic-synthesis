# Political Ad Analysis via LLMs

This repository contains code for analyzing political advertisements using large language models (LLMs). The analysis includes automatic annotation of ad content, topic classification, and stance analysis.

This codebase is designed to analyze political ads from Meta's Ad Library, focusing on ads related to U.S. politics. However, the methods and tools can be adapted for other datasets and contexts.

## Getting Started

To get started with the codebase, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing the Python environment.

```bash
uv sync
```

3. Prepare your dataset of ads. Upload your data to the `data/raw` directory as a CSV file.

4. Update the configuration file `config.yaml` with your dataset path and desired parameters. 

5. Run the scripts in the `scripts` directory to annotate and analyze the ads. Note: the models used in this codebase may require a GPU for efficient processing. vLLM is used for inference, so ensure you have a compatible setup (see [vLLM documentation](https://vllm.readthedocs.io/en/latest/installation.html) for details).

## Directory Structure

- `data/`: Contains raw and processed datasets.
- `scripts/`: Contains scripts for data processing, annotation, and analysis.
   - `0_preprocess.py`: Preprocess the raw ad data into a suitable format for annotation.
   - `1_embeddings.py`: Generate embeddings for the ad content using SentenceTransformers.
    - `2_generate_topics.py`: Synthesize a comprehensive list of topics using LLMs.
    - `3_annotate_clusters.py`: Utilize our 2-step LLM pipeline to annotate ad clusters with topics, stances, moral foundations, and summaries.
- `analysis/`: Jupyter notebooks for manual annotation and exploratory data analysis.
- `configs/`: Configuration files for different datasets and experiments. Note: Update these files with your dataset paths, models, and prompts.

## Configuration

The `config.yaml` file contains various parameters for the various scripts. The scripts use [Hydra](https://hydra.cc/docs/intro/) for configuration management, allowing you to easily switch between different settings. The configuration file includes settings for:
- Dataset paths
- Model specifications
- Prompt templates
