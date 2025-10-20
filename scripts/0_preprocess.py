"""Preprocess Meta Ads into a simplified format for LLM annotation."""

import json
from collections import defaultdict

import hydra
import pandas as pd
from omegaconf import DictConfig


def load_demographics(demographic_distribution: str) -> dict:
    """Load demographic distribution from JSON string.

    Args:
        demographic_distribution (str): JSON string of demographic distribution.

    Returns:
        dict:
            - most_common_age (str): Most common age group.
            - most_common_age_percentage (int): Percentage of most common age group.
            - most_common_gender (str): Most common gender.
            - most_common_gender_percentage (int): Percentage of most common gender.
    """
    ages = defaultdict(int)
    genders = defaultdict(int)
    demographics = json.loads(f"[{demographic_distribution}]")

    for entry in demographics:
        ages[entry["age"]] += entry["percentage"]

        if entry["gender"] != "unknown":
            genders[entry["gender"]] += entry["percentage"]

    most_common_age = max(ages.values())
    most_common_gender = max(genders.values())

    return {
        "most_common_age": most_common_age,
        "most_common_age_percentage": ages[most_common_age],
        "most_common_gender": most_common_gender,
        "most_common_gender_percentage": genders[most_common_gender],
    }


def get_avg_impressions(impressions: str) -> int:
    """Calculate average impressions from a range string.

    Args:
        impressions (str): Impressions range in the format 'lower_bound:1000,upper_bound:5000'.

    Returns:
        int: Average impressions.
    """
    impressions_range = {k.strip(): int(v) for k, v in (item.split(":") for item in impressions.split(","))}
    if "lower_bound" not in impressions_range:
        avg_impressions = impressions_range["upper_bound"]
    elif "upper_bound" not in impressions_range:
        avg_impressions = impressions_range["lower_bound"]
    else:
        avg_impressions = (impressions_range["lower_bound"] + impressions_range["upper_bound"]) // 2
    return avg_impressions


def split_demographics(impressions: str, demographic_distribution: str) -> dict:
    """
    Split demographic distribution into separate entries for each gender and age group.

    Args:
        impressions (str): Impressions range in the format 'lower_bound:1000,upper_bound:5000'.
        demographic_distribution (str): JSON string of demographic distribution.

    Returns:
        dict:
            Dictionary with keys:
            - gender: total_count
            - gender%: percentage
            - age: total_count
            - age%: percentage
            - gender_age: total_count
            - gender_age%: percentage
    """
    avg_impressions = get_avg_impressions(impressions)

    entries = dict()
    demographics = json.loads(f"[{demographic_distribution}]")
    gender = defaultdict(int)
    age = defaultdict(int)
    for entry in demographics:
        if entry["gender"] != "unknown" and entry["age"] != "Unknown":
            entries[entry["gender"] + "_" + entry["age"] + "%"] = entry["percentage"]
            entries[entry["gender"] + "_" + entry["age"]] = int(entry["percentage"] * avg_impressions)
        if entry["gender"] != "unknown":
            gender[entry["gender"]] += entry["percentage"]
        if entry["age"] != "Unknown":
            age[entry["age"]] += entry["percentage"]

    for k, v in gender.items():
        entries[k] = int(v * avg_impressions)
        entries[k + "%"] = v
    for k, v in age.items():
        entries[k] = int(v * avg_impressions)
        entries[k + "%"] = v

    return entries


def location_distribution(impressions: str, location_distribution: str) -> dict:
    """Process location distribution into state-level impressions.

    Args:
        impressions (str): Impressions range in the format 'lower_bound:1000,upper_bound:5000'.
        location_distribution (str): JSON string of location distribution.

    Returns:
        dict:
            Dictionary with keys:
            - state: total_count
            - state%: percentage
    """
    us_states = {
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
    }
    avg_impressions = get_avg_impressions(impressions)
    locations = json.loads(f"[{location_distribution}]")
    states = {
        location["region"]: int(location["percentage"]) for location in locations if location["region"] in us_states
    }

    netto_states = {k: v * avg_impressions for k, v in states.items()}
    percentage_states = {k + "%": int(v) for k, v in states.items()}
    netto_states.update(percentage_states)

    return netto_states


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Preprocess raw data to simplified format for LLM annotation."""
    # Load raw data
    file_path = cfg.data.dir.raw + "/" + cfg.data.filename
    df = pd.read_csv(file_path)

    # Remove ads with no demographic information
    df = df.dropna(subset=["demographic_distribution"])
    df = df.reset_index(drop=True)

    # Load primary demographics for each ad
    demographics = df["demographic_distribution"].apply(load_demographics)
    df = pd.concat([df, pd.DataFrame(demographics.tolist())], axis=1)

    # Get average impressions for each ad
    df["avg_impressions"] = df["impressions"].apply(get_avg_impressions)

    # Split demographics into separate entries
    split_demos = df.apply(lambda row: split_demographics(row["impressions"], row["demographic_distribution"]), axis=1)
    df = pd.concat([df, pd.DataFrame(split_demos.tolist())], axis=1)

    # Process location distribution into state-level impressions
    state_dists = df.apply(lambda row: location_distribution(row["impressions"], row["delivery_by_region"]), axis=1)
    df = pd.concat([df, pd.DataFrame(state_dists.tolist())], axis=1)

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Save preprocessed data
    save_path = cfg.data.dir.processed + "/" + cfg.data.filename
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()  # type: ignore
