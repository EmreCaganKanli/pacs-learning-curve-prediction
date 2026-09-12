import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CASE_FEATURES_CSV,
    COLD_START_DATASET_CSV,
    CURVE_TARGETS_CSV,
    DEVICE,
    SOURCE_CSV,
)
from data import build_data_state, make_loader
from metrics import (calculate_distribution_features, extract_features,)
from models import build_model


def main():
    if not CURVE_TARGETS_CSV.exists():
        raise FileNotFoundError("Run stages/03_build_curve_targets.py first.")

    if not SOURCE_CSV.exists():
        raise FileNotFoundError("Run stages/01_train_sources.py first.")

    data_state = build_data_state()
    splits = data_state["splits"]
    target_subsets = data_state["target_subsets"]

    curve_df = pd.read_csv(CURVE_TARGETS_CSV)

    source_df = pd.read_csv(SOURCE_CSV)

    source_lookup = (source_df.drop_duplicates("source_id").set_index("source_id"))

    if CASE_FEATURES_CSV.exists():
        existing_feature_df = pd.read_csv(CASE_FEATURES_CSV)

        redundant_columns = [
            "target_pool_size_5",
            "target_pool_size_10",
            "source_feature_count",
            "target_feature_count_5",
            "target_feature_count_10",
            "source_validation_accuracy",
        ]

        existing_feature_df = existing_feature_df.drop(
            columns=[
                column
                for column in redundant_columns
                if column in existing_feature_df.columns
            ]
        )

        completed_case_ids = set(existing_feature_df["case_id"].astype(str))

        feature_rows = (existing_feature_df.to_dict("records"))

        print(
            "Resuming with "
            f"{len(completed_case_ids)} "
            "completed cases."
        )

    else:
        completed_case_ids = set()
        feature_rows = []

    total_cases = len(curve_df)
    completed_counter = len(completed_case_ids)

    for source_id, source_cases in (curve_df.groupby("source_id")):
        source_row = source_lookup.loc[source_id]

        model_name = str(source_row["model"])

        source_seed = int(source_row["seed"])

        source_domains = tuple(str(source_row["sources"]).split("|"))

        checkpoint_path = Path(source_row["checkpoint"])

        cases_still_needed = (
            source_cases[~ source_cases["case_id"].astype(str).isin(completed_case_ids)]
        )

        if cases_still_needed.empty:
            continue

        print("\nLoading source model:", source_id,)

        model = build_model(model_name)

        model.load_state_dict(
            torch.load(checkpoint_path, map_location = "cpu", weights_only = True,)
        )

        model = model.to(DEVICE)

        combined_source_indices = (
            np.concatenate([splits[domain]["train"] for domain in source_domains])
        )

        source_loader = make_loader(combined_source_indices, train = False,)

        source_features, _ = (extract_features(model, source_loader,))

        target_feature_cache = {}

        for _, case in (cases_still_needed.iterrows()):
            case_id = str(case["case_id"])
            target_domain = str(case["target"])

            target_features_by_fraction = {}

            for target_fraction in [0.05, 0.10,]:
                cache_key = (target_domain, target_fraction,)

                if cache_key not in target_feature_cache:
                    target_indices = (target_subsets[target_domain][target_fraction])

                    target_loader = make_loader(target_indices, train = False,)

                    target_features, _ = (extract_features(model, target_loader,))

                    target_feature_cache[cache_key] = target_features

                target_features_by_fraction[target_fraction] = target_feature_cache[cache_key]

            distribution_metrics_by_fraction = {}

            for target_fraction, suffix in [(0.05, "5"), (0.10, "10"),]:
                fraction_metrics = (
                    calculate_distribution_features(
                        source_features,
                        target_features_by_fraction[target_fraction],
                        (
                            f"{case_id}_"
                            f"fraction{suffix}"
                        ),
                    )
                )

                distribution_metrics_by_fraction[target_fraction] = {
                    f"{metric_name}_{suffix}": metric_value
                    for metric_name, metric_value
                    in fraction_metrics.items()
                }

            feature_row = {
                "case_id": case_id,
                "source_id": source_id,
                "model": model_name,
                "sources": "|".join(source_domains),
                "source_count": len(source_domains),
                "target": target_domain,
                "seed": source_seed,
                "parameter_count": int(source_row["parameter_count"]),
                "source_size": int(len(combined_source_indices)),
                "feature_dimension": int(source_features.shape[1]),
            }

            feature_row.update(distribution_metrics_by_fraction[0.05])

            feature_row.update(distribution_metrics_by_fraction[0.10])

            feature_rows.append(feature_row)
            completed_case_ids.add(case_id)
            completed_counter += 1

            feature_df = pd.DataFrame(feature_rows)

            temporary_path = (CASE_FEATURES_CSV.with_suffix(".tmp.csv"))

            feature_df.to_csv(temporary_path, index = False,)

            temporary_path.replace(CASE_FEATURES_CSV)

            print(
                f"[{completed_counter}/"
                f"{total_cases}] "
                f"{case_id} | "
                f"MMD² 5%="
                f"{distribution_metrics_by_fraction[0.05]['mmd_rbf_squared_5']:.6f} | "
                f"MMD² 10%="
                f"{distribution_metrics_by_fraction[0.10]['mmd_rbf_squared_10']:.6f}"
            )

        del model
        del source_features
        del target_feature_cache

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    case_features_df = pd.DataFrame(feature_rows)

    case_features_df = (
        case_features_df
        .drop_duplicates(subset = ["case_id"], keep = "last",)
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    case_features_df.to_csv(CASE_FEATURES_CSV, index = False,)

    if not case_features_df["case_id"].is_unique:
        raise ValueError("case_features.csv must contain " "one row per case_id.")

    if set(case_features_df["case_id"]) != set(curve_df["case_id"]):
        raise ValueError("Case-feature IDs do not match " "curve-target IDs.")

    modeling_df = curve_df.merge(
        case_features_df,
        on=[
            "case_id",
            "source_id",
            "model",
            "sources",
            "source_count",
            "target",
            "seed",
            "parameter_count",
            "source_size",
        ],
        how="inner",
        validate="one_to_one",
    )

    modeling_df.to_csv(COLD_START_DATASET_CSV, index = False,)

    print("\nSaved cold-start features to:", CASE_FEATURES_CSV,)

    print("Saved prediction dataset to:", COLD_START_DATASET_CSV,)


if __name__ == "__main__":
    main()