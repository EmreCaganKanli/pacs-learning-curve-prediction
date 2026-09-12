import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from config import (
    CURVE_PARAMETER_NAMES,
    RESULT_PLOT_DIR,
    RIDGE_ALPHAS,
    WARM_START_DATASET_CSV,
    WARM_START_FOLD_SUMMARY_CSV,
    WARM_START_PREDICTIONS_CSV,
    WARM_START_SUMMARY_CSV,
)
from data import build_data_state
from prediction import (
    build_ridge_pipeline,
    reconstruct_power_law_curves,
    sanitize_curve_parameters,
    select_ridge_alpha,
    three_point_log_curve_predictions,
)

import sklearn

print("Python:", sys.executable)
print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)

def fit_ridge_variant(
    train_df,
    test_df,
    y_train,
    numeric_features,
    categorical_features,
    feature_columns,
    prediction_n_columns,
    prediction_accuracy_columns,
):
    best_alpha = select_ridge_alpha(
        train_df=train_df,
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_columns=feature_columns,
        alpha_values=RIDGE_ALPHAS,
        n_columns=prediction_n_columns,
        accuracy_columns=prediction_accuracy_columns,
    )

    pipeline = build_ridge_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        alpha=best_alpha,
    )

    pipeline.fit(train_df[feature_columns], y_train,)

    parameter_predictions = pipeline.predict(test_df[feature_columns])

    curve_predictions = reconstruct_power_law_curves(
        test_df,
        parameter_predictions,
        prediction_n_columns,
    )

    return (curve_predictions, parameter_predictions, best_alpha,)


def main():
    if not WARM_START_DATASET_CSV.exists():
        raise FileNotFoundError("Run stages/06_build_warm_start_dataset.py first.")

    domains = build_data_state()["domains"]

    warm_modeling_df = pd.read_csv(WARM_START_DATASET_CSV)

    target_columns = CURVE_PARAMETER_NAMES

    prediction_n_columns = ["n_25", "n_50", "n_100",]

    prediction_accuracy_columns = ["accuracy_25", "accuracy_50", "accuracy_100",]

    prediction_points = [25, 50, 100,]

    warm_categorical_features = ["model",]

    metadata_numeric_features = [
        "source_size",
        "n_5",
        "source_accuracy",
        "zero_shot_val_accuracy",
        "zero_shot_val_loss",
    ]

    source_distribution_features = [
        "mmd_rbf_squared_10",
        "centroid_l2_10",
        "centroid_cosine_distance_10",
        "covariance_frobenius_normalized_10",
    ]

    adapted_distribution_features = [
        "adapted_mmd_rbf_squared_10",
        "adapted_centroid_l2_10",
        "adapted_centroid_cosine_distance_10",
        "adapted_covariance_frobenius_normalized_10",
    ]

    delta_distribution_features = [
        "delta_mmd_rbf_squared_10",
        "delta_centroid_l2_10",
        "delta_centroid_cosine_distance_10",
        "delta_covariance_frobenius_normalized_10",
    ]

    observed_adaptation_features = [
        "n_10",

        "best_epoch_5",
        "best_train_accuracy_5",
        "best_val_accuracy_5",
        "optimization_steps_5",

        "best_epoch_10",
        "best_train_accuracy_10",
        "best_val_accuracy_10",
        "optimization_steps_10",
    ]

    observation_only_features = (observed_adaptation_features)

    metadata_observation_numeric_features = (
        metadata_numeric_features
        + observed_adaptation_features
    )

    metadata_observation_features = (
        warm_categorical_features
        + metadata_observation_numeric_features
    )

    metadata_observation_source_numeric_features = (
        metadata_observation_numeric_features
        + source_distribution_features
    )

    metadata_observation_source_features = (
        warm_categorical_features
        + metadata_observation_source_numeric_features
    )

    metadata_observation_adapted_numeric_features = (
        metadata_observation_numeric_features
        + adapted_distribution_features
    )

    metadata_observation_adapted_features = (
        warm_categorical_features
        + metadata_observation_adapted_numeric_features
    )

    metadata_observation_change_numeric_features = (
        metadata_observation_numeric_features
        + delta_distribution_features
    )

    metadata_observation_change_features = (
        warm_categorical_features
        + metadata_observation_change_numeric_features
    )

    metadata_observation_adapted_change_numeric_features = (
        metadata_observation_numeric_features
        + adapted_distribution_features
        + delta_distribution_features
    )

    metadata_observation_adapted_change_features = (
        warm_categorical_features
        + metadata_observation_adapted_change_numeric_features
    )

    ridge_variants = {
        "observation_ridge": {
            "numeric": observed_adaptation_features,
            "categorical": [],
            "features": observation_only_features,
        },
        "metadata_observation_ridge": {
            "numeric": metadata_observation_numeric_features,
            "categorical": warm_categorical_features,
            "features": metadata_observation_features,
        },
        "metadata_observation_source_ridge": {
            "numeric": metadata_observation_source_numeric_features,
            "categorical": warm_categorical_features,
            "features": metadata_observation_source_features,
        },
        "metadata_observation_adapted_ridge": {
            "numeric": metadata_observation_adapted_numeric_features,
            "categorical": warm_categorical_features,
            "features": metadata_observation_adapted_features,
        },
        "metadata_observation_change_ridge": {
            "numeric": metadata_observation_change_numeric_features,
            "categorical": warm_categorical_features,
            "features": metadata_observation_change_features,
        },
        "metadata_observation_adapted_change_ridge": {
            "numeric": (metadata_observation_adapted_change_numeric_features),
            "categorical": warm_categorical_features,
            "features": (metadata_observation_adapted_change_features),
        },
    }

    required_columns = list(
        dict.fromkeys(
            ["case_id", "source_id", "model", "sources", "target", "seed", "n_5", "n_10",]
            + target_columns
            + prediction_n_columns
            + prediction_accuracy_columns
            + metadata_observation_features
            + source_distribution_features
            + adapted_distribution_features
            + delta_distribution_features
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in warm_modeling_df.columns
    ]

    if missing_columns:
        raise ValueError("Missing warm-start columns: " + ", ".join(missing_columns))

    missing_counts = (warm_modeling_df[required_columns].isna().sum())

    if (missing_counts > 0).any():
        raise ValueError("Missing warm-start values:\n" + str(missing_counts[missing_counts > 0]))

    y = warm_modeling_df[target_columns].to_numpy(dtype=float)

    warm_prediction_rows = []
    warm_fold_summary_rows = []

    model_names = ["mean_remaining_curve", "three_point_log_curve", * ridge_variants.keys(),]

    for fold_number, held_out_target in enumerate(domains, start = 1,):
        train_indices = warm_modeling_df.index[
            warm_modeling_df["target"]
            != held_out_target
        ].to_numpy()

        test_indices = warm_modeling_df.index[
            warm_modeling_df["target"]
            == held_out_target
        ].to_numpy()

        train_df = warm_modeling_df.iloc[train_indices].copy()

        test_df = warm_modeling_df.iloc[test_indices].copy()

        y_train = y[train_indices]

        actual_curves = test_df[prediction_accuracy_columns].to_numpy(dtype=float)

        mean_parameters = y_train.mean(axis = 0, keepdims = True,)

        mean_parameter_predictions = np.repeat(mean_parameters, repeats = len(test_df), axis = 0,)

        mean_predictions = reconstruct_power_law_curves(
            test_df,
            mean_parameter_predictions,
            prediction_n_columns,
        )

        prediction_n = test_df[prediction_n_columns].to_numpy(dtype=float)

        log_curve_predictions = (
            three_point_log_curve_predictions(
                zero_shot_accuracy=(test_df["zero_shot_val_accuracy"].to_numpy()),
                first_n=(test_df["n_5"].to_numpy()),
                first_accuracy=(test_df["best_val_accuracy_5"].to_numpy()),
                second_n=(test_df["n_10"].to_numpy()),
                second_accuracy=(test_df["best_val_accuracy_10"].to_numpy()),
                prediction_n=prediction_n,
            )
        )

        fold_predictions = {
            "mean_remaining_curve": mean_predictions,
            "three_point_log_curve": log_curve_predictions,
        }

        fold_parameter_predictions = {"mean_remaining_curve": (mean_parameter_predictions),}

        selected_alphas = {"mean_remaining_curve": np.nan, "three_point_log_curve": np.nan,}

        for model_name, variant in ridge_variants.items():
            (curve_predictions, parameter_predictions, best_alpha,) = fit_ridge_variant(
                train_df=train_df,
                test_df=test_df,
                y_train=y_train,
                numeric_features=variant["numeric"],
                categorical_features=variant["categorical"],
                feature_columns=variant["features"],
                prediction_n_columns=(prediction_n_columns),
                prediction_accuracy_columns=(prediction_accuracy_columns),
            )
            fold_predictions[model_name] = curve_predictions

            fold_parameter_predictions[model_name] = parameter_predictions

            selected_alphas[model_name] = best_alpha

        for model_name, predictions in (fold_predictions.items()):
            if model_name in fold_parameter_predictions:
                parameter_predictions = (
                    sanitize_curve_parameters(fold_parameter_predictions[model_name])
                )
            else:
                parameter_predictions = None

            for (local_position, dataframe_index,) in enumerate(test_indices):
                row = warm_modeling_df.iloc[dataframe_index]

                result_row = {
                    "fold": fold_number,
                    "held_out_target": (held_out_target),
                    "prediction_model": (model_name),
                    "selected_alpha": (selected_alphas[model_name]),
                    "case_id": row["case_id"],
                    "source_id": row["source_id"],
                    "model": row["model"],
                    "sources": row["sources"],
                    "target": row["target"],
                    "seed": int(row["seed"]),
                    "observed_zero_shot_val_accuracy": float(row["zero_shot_val_accuracy"]),
                    "observed_best_val_accuracy_5": float(row["best_val_accuracy_5"]),
                    "observed_best_val_accuracy_10": float(row["best_val_accuracy_10"]),
                }

                if parameter_predictions is not None:
                    for (parameter_position, parameter_name,) in enumerate(target_columns):
                        result_row[
                            f"actual_{parameter_name}"
                        ] = float(row[parameter_name])

                        result_row[
                            f"predicted_{parameter_name}"
                        ] = float(parameter_predictions[local_position, parameter_position,])

                for (point_position, point,) in enumerate(prediction_points):
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

                warm_prediction_rows.append(result_row)

        for model_name, predictions in (fold_predictions.items()):
            point_maes = {
                point: mean_absolute_error(actual_curves[:, position], predictions[:, position],)
                for position, point in enumerate(prediction_points)
            }

            fold_row = {
                "fold": fold_number,
                "held_out_target": (held_out_target),
                "prediction_model": model_name,
                "selected_alpha": (selected_alphas[model_name]),
            }

            for point in prediction_points:
                fold_row[
                    f"mae_{point}"
                ] = point_maes[point]

            fold_row["overall_mae"] = float(np.mean(list(point_maes.values())))

            warm_fold_summary_rows.append(fold_row)

    predictions_df = pd.DataFrame(warm_prediction_rows)

    predictions_df.to_csv(WARM_START_PREDICTIONS_CSV, index = False,)

    fold_summary_df = (
        pd.DataFrame(warm_fold_summary_rows)
        .sort_values(["held_out_target", "overall_mae",])
        .reset_index(drop=True)
    )

    fold_summary_df.to_csv(WARM_START_FOLD_SUMMARY_CSV, index = False,)

    summary_rows = []

    for model_name in model_names:
        model_predictions = (predictions_df[predictions_df["prediction_model"] == model_name])

        point_maes = {
            point: mean_absolute_error(
                model_predictions[
                    f"actual_{point}"
                ],
                model_predictions[
                    f"predicted_{point}"
                ],
            )
            for point in prediction_points
        }

        summary_row = {"prediction_model": model_name,}

        for point in prediction_points:
            summary_row[
                f"mae_{point}"
            ] = point_maes[point]

        summary_row["overall_mae"] = float(np.mean(list(point_maes.values())))

        summary_rows.append(summary_row)

    summary_df = (pd.DataFrame(summary_rows).sort_values("overall_mae").reset_index(drop = True))

    summary_df.to_csv(WARM_START_SUMMARY_CSV, index = False,)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(prediction_points))

    for model_name in model_names:
        row = summary_df[summary_df["prediction_model"] == model_name].iloc[0]

        plt.plot(
            x_positions,
            [
                row[f"mae_{point}"]
                for point in prediction_points
            ],
            marker="o",
            label=model_name,
        )

    plt.xticks(
        x_positions,
        [
            f"{point}%"
            for point in prediction_points
        ],
    )

    plt.xlabel("Target training budget")

    plt.ylabel("Mean absolute error")

    plt.title("Warm-start prediction after " "observing the first two adaptation budgets")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULT_PLOT_DIR / "warm_start_mae_by_budget.png", dpi = 150, bbox_inches = "tight",)

    plt.close()

    print("\nWarm-start summary:\n", summary_df.to_string(index = False),)


if __name__ == "__main__":
    main()
