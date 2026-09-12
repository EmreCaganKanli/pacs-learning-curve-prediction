import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COLD_START_DATASET_CSV,
    COLD_START_FOLD_SUMMARY_CSV,
    COLD_START_PREDICTIONS_CSV,
    COLD_START_SUMMARY_CSV,
    CURVE_PARAMETER_NAMES,
    RESULT_PLOT_DIR,
    RIDGE_ALPHAS,
)
from data import build_data_state
from prediction import (
    build_ridge_pipeline,
    reconstruct_power_law_curves,
    sanitize_curve_parameters,
    select_ridge_alpha,
)


def main():
    if not COLD_START_DATASET_CSV.exists():
        raise FileNotFoundError("Run stages/04_extract_cold_start_features.py first.")

    domains = build_data_state()["domains"]

    modeling_df = pd.read_csv(COLD_START_DATASET_CSV)

    target_columns = CURVE_PARAMETER_NAMES

    n_columns = ["n_0", "n_5", "n_10", "n_25", "n_50", "n_100",]

    accuracy_columns = [
        "accuracy_0",
        "accuracy_5",
        "accuracy_10",
        "accuracy_25",
        "accuracy_50",
        "accuracy_100",
    ]

    curve_points = [0, 5, 10, 25, 50, 100,]

    categorical_features = ["model"]

    metadata_numeric_features = [
        "source_size",
        "n_5",
        "source_accuracy",
        "zero_shot_val_accuracy",
        "zero_shot_val_loss",
    ]

    distribution_features = [
        "mmd_rbf_squared_5",
        "centroid_l2_5",
        "centroid_cosine_distance_5",
        "covariance_frobenius_normalized_5",
    ]

    metadata_features = (categorical_features + metadata_numeric_features)

    full_numeric_features = (metadata_numeric_features + distribution_features)

    full_features = (categorical_features + full_numeric_features)

    required_columns = list(
        dict.fromkeys(
            ["case_id", "source_id", "model", "sources", "target", "seed",]
            + target_columns
            + n_columns
            + accuracy_columns
            + metadata_features
            + distribution_features
        )
    )

    missing_columns = [column for column in required_columns if column not in modeling_df.columns]

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    missing_counts = (modeling_df[required_columns].isna().sum())

    if (missing_counts > 0).any():
        raise ValueError("Missing values found:\n" + str(missing_counts[missing_counts > 0]))

    y = modeling_df[target_columns].to_numpy(dtype=float)

    prediction_rows = []
    fold_summary_rows = []

    prediction_model_names = [
        "mean_curve",
        "metadata_ridge",
        "distance_ridge",
        "metadata_distance_ridge",
    ]

    for fold_number, held_out_target in enumerate(domains, start = 1,):
        train_indices = modeling_df.index[modeling_df["target"] != held_out_target].to_numpy()

        test_indices = modeling_df.index[modeling_df["target"] == held_out_target].to_numpy()

        train_df = modeling_df.iloc[train_indices].copy()

        test_df = modeling_df.iloc[test_indices].copy()

        y_train = y[train_indices]

        actual_curves = test_df[accuracy_columns].to_numpy(dtype=float)

        mean_parameters = y_train.mean(axis = 0, keepdims = True,)

        mean_parameter_predictions = np.repeat(mean_parameters, repeats = len(test_df), axis = 0,)

        mean_parameter_predictions = sanitize_curve_parameters(mean_parameter_predictions)

        mean_predictions = reconstruct_power_law_curves(
            test_df,
            mean_parameter_predictions,
            n_columns,
        )

        best_metadata_alpha = select_ridge_alpha(
            train_df=train_df,
            y_train=y_train,
            numeric_features=metadata_numeric_features,
            categorical_features=categorical_features,
            feature_columns=metadata_features,
            alpha_values=RIDGE_ALPHAS,
            n_columns=n_columns,
            accuracy_columns=accuracy_columns,
        )

        metadata_pipeline = build_ridge_pipeline(
            numeric_features=metadata_numeric_features,
            categorical_features=categorical_features,
            alpha=best_metadata_alpha,
        )

        metadata_pipeline.fit(train_df[metadata_features], y_train,)

        metadata_parameter_predictions = (metadata_pipeline.predict(test_df[metadata_features]))

        metadata_parameter_predictions = sanitize_curve_parameters(metadata_parameter_predictions)

        metadata_predictions = reconstruct_power_law_curves(
            test_df,
            metadata_parameter_predictions,
            n_columns,
        )

        best_distance_alpha = select_ridge_alpha(
            train_df=train_df,
            y_train=y_train,
            numeric_features=distribution_features,
            categorical_features=[],
            feature_columns=distribution_features,
            alpha_values=RIDGE_ALPHAS,
            n_columns=n_columns,
            accuracy_columns=accuracy_columns,
        )

        distance_pipeline = build_ridge_pipeline(
            numeric_features=distribution_features,
            categorical_features=[],
            alpha=best_distance_alpha,
        )

        distance_pipeline.fit(train_df[distribution_features], y_train,)

        distance_parameter_predictions = (distance_pipeline.predict(test_df[distribution_features]))

        distance_parameter_predictions = sanitize_curve_parameters(distance_parameter_predictions)

        distance_predictions = reconstruct_power_law_curves(
            test_df,
            distance_parameter_predictions,
            n_columns,
        )

        best_full_alpha = select_ridge_alpha(
            train_df=train_df,
            y_train=y_train,
            numeric_features=full_numeric_features,
            categorical_features=categorical_features,
            feature_columns=full_features,
            alpha_values=RIDGE_ALPHAS,
            n_columns=n_columns,
            accuracy_columns=accuracy_columns,
        )

        full_pipeline = build_ridge_pipeline(
            numeric_features=full_numeric_features,
            categorical_features=categorical_features,
            alpha=best_full_alpha,
        )

        full_pipeline.fit(train_df[full_features], y_train,)

        full_parameter_predictions = (full_pipeline.predict(test_df[full_features]))

        full_parameter_predictions = sanitize_curve_parameters(full_parameter_predictions)

        full_predictions = reconstruct_power_law_curves(
            test_df,
            full_parameter_predictions,
            n_columns,
        )

        fold_predictions = {
            "mean_curve": mean_predictions,
            "metadata_ridge": metadata_predictions,
            "distance_ridge": distance_predictions,
            "metadata_distance_ridge": full_predictions,
        }

        fold_parameter_predictions = {
            "mean_curve": mean_parameter_predictions,
            "metadata_ridge": metadata_parameter_predictions,
            "distance_ridge": distance_parameter_predictions,
            "metadata_distance_ridge": full_parameter_predictions,
        }

        selected_alphas = {
            "mean_curve": np.nan,
            "metadata_ridge": best_metadata_alpha,
            "distance_ridge": best_distance_alpha,
            "metadata_distance_ridge": best_full_alpha,
        }

        for model_name, predictions in fold_predictions.items():
            parameter_predictions = fold_parameter_predictions[model_name]

            for local_position, dataframe_index in enumerate(test_indices):
                row = modeling_df.iloc[dataframe_index]

                result_row = {
                    "fold": fold_number,
                    "held_out_target": held_out_target,
                    "prediction_model": model_name,
                    "selected_alpha": selected_alphas[model_name],
                    "case_id": row["case_id"],
                    "source_id": row["source_id"],
                    "model": row["model"],
                    "sources": row["sources"],
                    "target": row["target"],
                    "seed": int(row["seed"]),
                }

                for parameter_position, parameter_name in enumerate(target_columns):
                    result_row[
                        f"actual_{parameter_name}"
                    ] = float(row[parameter_name])

                    result_row[
                        f"predicted_{parameter_name}"
                    ] = float(parameter_predictions[local_position, parameter_position,])

                for point_position, point in enumerate(curve_points):
                    actual = float(actual_curves[local_position, point_position,])

                    predicted = float(predictions[local_position, point_position,])

                    result_row[
                        f"n_{point}"
                    ] = int(
                        row[f"n_{point}"]
                    )

                    result_row[
                        f"actual_{point}"
                    ] = actual

                    result_row[
                        f"predicted_{point}"
                    ] = predicted

                    result_row[
                        f"absolute_error_{point}"
                    ] = abs(actual - predicted)

                prediction_rows.append(result_row)

        for model_name, predictions in fold_predictions.items():
            point_maes = {
                point: mean_absolute_error(actual_curves[:, position], predictions[:, position],)
                for position, point in enumerate(curve_points)
            }

            fold_row = {
                "fold": fold_number,
                "held_out_target": held_out_target,
                "prediction_model": model_name,
                "selected_alpha": selected_alphas[model_name],
            }

            for point in curve_points:
                fold_row[
                    f"mae_{point}"
                ] = point_maes[point]

            fold_row["overall_mae"] = float(np.mean(list(point_maes.values())))

            fold_summary_rows.append(fold_row)

    predictions_df = pd.DataFrame(prediction_rows)

    predictions_df.to_csv(COLD_START_PREDICTIONS_CSV, index = False,)

    fold_summary_df = (
        pd.DataFrame(fold_summary_rows)
        .sort_values(["held_out_target", "overall_mae",])
        .reset_index(drop=True)
    )

    fold_summary_df.to_csv(COLD_START_FOLD_SUMMARY_CSV, index = False,)

    summary_rows = []

    for model_name in prediction_model_names:
        model_predictions = predictions_df[predictions_df["prediction_model"] == model_name]

        point_maes = {
            point: mean_absolute_error(
                model_predictions[
                    f"actual_{point}"
                ],
                model_predictions[
                    f"predicted_{point}"
                ],
            )
            for point in curve_points
        }

        summary_row = {"prediction_model": model_name,}

        for point in curve_points:
            summary_row[
                f"mae_{point}"
            ] = point_maes[point]

        summary_row["overall_mae"] = float(np.mean(list(point_maes.values())))

        summary_rows.append(summary_row)

    summary_df = (pd.DataFrame(summary_rows).sort_values("overall_mae").reset_index(drop = True))

    summary_df.to_csv(COLD_START_SUMMARY_CSV, index = False,)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(curve_points))

    for model_name in prediction_model_names:
        row = summary_df[summary_df["prediction_model"] == model_name].iloc[0]

        plt.plot(
            x_positions,
            [
                row[f"mae_{point}"]
                for point in curve_points
            ],
            marker="o",
            label=model_name,
        )

    plt.xticks(
        x_positions,
        [
            "0"
            if point == 0
            else f"{point}%"
            for point in curve_points
        ],
    )

    plt.xlabel("Nominal target training budget")

    plt.ylabel("Mean absolute error")

    plt.title("Cold-start leave-one-target-domain-out")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULT_PLOT_DIR / "cold_start_mae_by_budget.png", dpi = 150, bbox_inches = "tight",)

    plt.close()

    print("\nCold-start summary:\n", summary_df.to_string(index = False),)


if __name__ == "__main__":
    main()