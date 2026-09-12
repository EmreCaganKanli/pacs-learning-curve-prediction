import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (ADAPTATION_CSV, COLD_START_DATASET_CSV, DEVICE, WARM_START_DATASET_CSV,)
from data import build_data_state, make_loader
from metrics import (calculate_distribution_features, extract_features,)
from models import build_model


CORE_DISTRIBUTION_METRICS = [
    "mmd_rbf_squared",
    "centroid_l2",
    "centroid_cosine_distance",
    "covariance_frobenius_normalized",
]


def main():
    if not ADAPTATION_CSV.exists():
        raise FileNotFoundError("Run stages/02_run_adaptation.py first.")

    if not COLD_START_DATASET_CSV.exists():
        raise FileNotFoundError("Run stages/04_extract_cold_start_features.py first.")

    data_state = build_data_state()
    splits = data_state["splits"]
    target_subsets = data_state["target_subsets"]

    adaptation_df = pd.read_csv(ADAPTATION_CSV)

    cold_start_df = pd.read_csv(COLD_START_DATASET_CSV)

    required_cold_start_columns = [
        "case_id",
        "model",
        "sources",
        "target",
        "zero_shot_val_accuracy",
        "zero_shot_val_loss",
        "n_5",
        "n_10",
        "mmd_rbf_squared_5",
        "centroid_l2_5",
        "centroid_cosine_distance_5",
        "covariance_frobenius_normalized_5",
        "mmd_rbf_squared_10",
        "centroid_l2_10",
        "centroid_cosine_distance_10",
        "covariance_frobenius_normalized_10",
    ]

    missing_columns = [
        column
        for column in required_cold_start_columns
        if column not in cold_start_df.columns
    ]

    if missing_columns:
        raise ValueError("Missing required cold-start columns: " + ", ".join(missing_columns))

    required_adaptation_columns = [
        "case_id",
        "fraction_percent",
        "adapted_checkpoint",
        "best_epoch",
        "best_train_accuracy",
        "best_val_accuracy",
        "optimization_steps",
    ]

    missing_columns = [
        column
        for column in required_adaptation_columns
        if column not in adaptation_df.columns
    ]

    if missing_columns:
        raise ValueError("Missing required adaptation columns: " + ", ".join(missing_columns))

    five_percent_df = adaptation_df[adaptation_df["fraction_percent"] == 5].copy()

    ten_percent_df = adaptation_df[adaptation_df["fraction_percent"] == 10].copy()

    if not five_percent_df["case_id"].is_unique:
        raise ValueError("5% adaptation rows are not unique by case_id.")

    if not ten_percent_df["case_id"].is_unique:
        raise ValueError("10% adaptation rows are not unique by case_id.")

    if len(five_percent_df) != len(cold_start_df):
        raise ValueError("The 5% rows do not match the cold-start cases.")

    if len(ten_percent_df) != len(cold_start_df):
        raise ValueError("The 10% rows do not match the cold-start cases.")

    five_percent_df = five_percent_df.rename(
        columns={
            "best_epoch": "best_epoch_5",
            "best_train_accuracy": "best_train_accuracy_5",
            "best_val_accuracy": "best_val_accuracy_5",
            "optimization_steps": "optimization_steps_5",
        }
    )

    ten_percent_df = ten_percent_df.rename(
        columns={
            "best_epoch": "best_epoch_10",
            "best_train_accuracy": "best_train_accuracy_10",
            "best_val_accuracy": "best_val_accuracy_10",
            "optimization_steps": "optimization_steps_10",
        }
    )

    five_percent_columns = [
        "case_id",
        "best_epoch_5",
        "best_train_accuracy_5",
        "best_val_accuracy_5",
        "optimization_steps_5",
    ]

    ten_percent_columns = [
        "case_id",
        "adapted_checkpoint",
        "best_epoch_10",
        "best_train_accuracy_10",
        "best_val_accuracy_10",
        "optimization_steps_10",
    ]

    warm_start_df = (
        cold_start_df
        .merge(
            five_percent_df[five_percent_columns],
            on="case_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            ten_percent_df[ten_percent_columns],
            on="case_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(warm_start_df) != len(cold_start_df):
        raise ValueError("Warm-start merge lost cases.")

    if not warm_start_df["case_id"].is_unique:
        raise ValueError("Warm-start dataset must contain " "one row per case_id.")

    new_feature_columns = []

    for metric_name in CORE_DISTRIBUTION_METRICS:
        new_feature_columns.extend([
            f"adapted_{metric_name}_10",
            f"delta_{metric_name}_10",
        ])

    adapted_feature_rows = []
    completed_case_ids = set()

    if WARM_START_DATASET_CSV.exists():
        existing_df = pd.read_csv(WARM_START_DATASET_CSV)

        if all(column in existing_df.columns for column in new_feature_columns):
            existing_features = (
                existing_df[["case_id"] + new_feature_columns]
                .dropna(subset=new_feature_columns)
                .drop_duplicates("case_id")
            )

            adapted_feature_rows = (existing_features.to_dict("records"))

            completed_case_ids = set(existing_features["case_id"].astype(str))

    print("Completed adapted-feature cases:", len(completed_case_ids),)

    total_cases = len(warm_start_df)

    for case_number, row in warm_start_df.iterrows():
        case_id = str(row["case_id"])

        if case_id in completed_case_ids:
            print(
                f"[{case_number + 1}/{total_cases}] "
                f"Skipping {case_id}"
            )
            continue

        model_name = str(row["model"])
        target_domain = str(row["target"])

        source_domains = tuple(str(row["sources"]).split("|"))

        checkpoint_path = Path(row["adapted_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing 10% adapted checkpoint: "
                f"{checkpoint_path}"
            )

        print(
            f"[{case_number + 1}/{total_cases}] "
            f"{case_id}"
        )

        model = build_model(model_name)

        model.load_state_dict(
            torch.load(checkpoint_path, map_location = "cpu", weights_only = True,)
        )

        model = model.to(DEVICE)

        combined_source_indices = np.concatenate([
            splits[domain]["train"]
            for domain in source_domains
        ])

        target_indices = (target_subsets[target_domain][0.10])

        source_loader = make_loader(combined_source_indices, train = False,)

        target_loader = make_loader(target_indices, train = False,)

        source_features, _ = extract_features(model, source_loader,)

        target_features, _ = extract_features(model, target_loader,)

        adapted_metrics = (
            calculate_distribution_features(
                source_features,
                target_features,
                f"{case_id}_adapted_fraction10",
            )
        )

        feature_row = {"case_id": case_id,}

        for metric_name in CORE_DISTRIBUTION_METRICS:
            adapted_column = (
                f"adapted_{metric_name}_10"
            )

            original_column = (
                f"{metric_name}_10"
            )

            delta_column = (
                f"delta_{metric_name}_10"
            )

            adapted_value = float(adapted_metrics[metric_name])

            original_value = float(row[original_column])

            feature_row[adapted_column] = (adapted_value)

            feature_row[delta_column] = (adapted_value - original_value)

        adapted_feature_rows.append(feature_row)

        completed_case_ids.add(case_id)

        partial_feature_df = pd.DataFrame(adapted_feature_rows)

        partial_warm_df = warm_start_df.merge(
            partial_feature_df,
            on="case_id",
            how="left",
            validate="one_to_one",
        )

        temporary_path = (WARM_START_DATASET_CSV.with_suffix(".tmp.csv"))

        partial_warm_df.to_csv(temporary_path, index = False,)

        temporary_path.replace(WARM_START_DATASET_CSV)

        del model
        del source_features
        del target_features

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adapted_feature_df = pd.DataFrame(adapted_feature_rows)

    if not adapted_feature_df["case_id"].is_unique:
        raise ValueError("Adapted discrepancy rows must be unique " "by case_id.")

    warm_start_df = warm_start_df.merge(
        adapted_feature_df,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    if len(warm_start_df) != len(cold_start_df):
        raise ValueError("Adapted-feature merge lost cases.")

    if warm_start_df[new_feature_columns].isna().any().any():
        raise ValueError("Missing adapted warm-start " "discrepancy values.")

    warm_start_df.to_csv(WARM_START_DATASET_CSV, index = False,)

    print("Saved warm-start dataset to:", WARM_START_DATASET_CSV,)

    print("Warm-start dataset shape:", warm_start_df.shape,)


if __name__ == "__main__":
    main()