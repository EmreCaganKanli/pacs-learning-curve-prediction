import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COLD_LOFO_ABLATION_CSV,
    COLD_RIDGE_COEFFICIENTS_CSV,
    COLD_RIDGE_COEFFICIENT_SUMMARY_CSV,
    COLD_RIDGE_PARAMETER_COEFFICIENTS_CSV,
    COLD_START_DATASET_CSV,
    COLD_START_FOLD_SUMMARY_CSV,
    CURVE_PARAMETER_NAMES,
    RIDGE_ALPHAS,
    WARM_FEATURE_CORRELATION_MATRIX_CSV,
    WARM_FEATURE_CORRELATION_PAIRS_CSV,
    WARM_GROUPED_ABLATION_CSV,
    WARM_RIDGE_COEFFICIENTS_CSV,
    WARM_RIDGE_COEFFICIENT_SUMMARY_CSV,
    WARM_RIDGE_PARAMETER_COEFFICIENTS_CSV,
    WARM_START_DATASET_CSV,
    WARM_START_FOLD_SUMMARY_CSV,
)
from prediction import build_ridge_pipeline, reconstruct_power_law_curves, select_ridge_alpha
from training import atomic_save_csv


CATEGORICAL_FEATURES = ["model"]

COLD_NUMERIC_FEATURES = [
    "source_size",
    "n_5",
    "source_accuracy",
    "zero_shot_val_accuracy",
    "zero_shot_val_loss",
]

WARM_OBSERVATION_FEATURES = [
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
WARM_NUMERIC_FEATURES = COLD_NUMERIC_FEATURES + WARM_OBSERVATION_FEATURES

COLD_N_COLUMNS = ["n_0", "n_5", "n_10", "n_25", "n_50", "n_100"]
COLD_ACCURACY_COLUMNS = ["accuracy_0", "accuracy_5", "accuracy_10", "accuracy_25", "accuracy_50", "accuracy_100"]
WARM_N_COLUMNS = ["n_25", "n_50", "n_100"]
WARM_ACCURACY_COLUMNS = ["accuracy_25", "accuracy_50", "accuracy_100"]

WARM_FEATURE_GROUPS = {
    "Source / zero-shot context": {
        "numeric": ["source_size", "source_accuracy", "zero_shot_val_accuracy", "zero_shot_val_loss"],
        "categorical": ["model"],
    },
    "Budget / training amount": {
        "numeric": ["n_5", "n_10", "optimization_steps_5", "optimization_steps_10"],
        "categorical": [],
    },
    "Training dynamics": {
        "numeric": ["best_epoch_5", "best_epoch_10", "best_train_accuracy_5", "best_train_accuracy_10"],
        "categorical": [],
    },
    "Early validation performance": {
        "numeric": ["best_val_accuracy_5", "best_val_accuracy_10"],
        "categorical": [],
    },
}


def require_columns(dataframe, columns, label):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: " + ", ".join(missing))


def ridge_numeric_coefficients_by_fold(
    modeling_df,
    fold_summary_df,
    prediction_model,
    numeric_features,
    categorical_features,
):
    rows = []

    for held_out_target in sorted(modeling_df["target"].unique()):
        train_df = modeling_df[modeling_df["target"] != held_out_target].copy()
        alpha_row = fold_summary_df[
            (fold_summary_df["held_out_target"] == held_out_target)
            & (fold_summary_df["prediction_model"] == prediction_model)
        ]

        if len(alpha_row) != 1:
            raise ValueError(f"Expected one selected alpha for {prediction_model} / {held_out_target}.")

        alpha = float(alpha_row.iloc[0]["selected_alpha"])
        feature_columns = categorical_features + numeric_features
        y_train = train_df[CURVE_PARAMETER_NAMES].to_numpy(dtype=float)

        pipeline = build_ridge_pipeline(numeric_features, categorical_features, alpha)
        pipeline.fit(train_df[feature_columns], y_train)

        coefficients = pipeline.named_steps["ridge"].coef_
        target_std = y_train.std(axis=0, ddof=0)
        target_std[target_std == 0] = 1.0
        standardized_coefficients = coefficients[:, :len(numeric_features)] / target_std[:, None]

        for parameter_index, parameter_name in enumerate(CURVE_PARAMETER_NAMES):
            for feature_index, feature_name in enumerate(numeric_features):
                rows.append({
                    "held_out_target": held_out_target,
                    "selected_alpha": alpha,
                    "parameter": parameter_name,
                    "feature": feature_name,
                    "standardized_coefficient": float(standardized_coefficients[parameter_index, feature_index]),
                })

    return pd.DataFrame(rows)


def summarize_ridge_coefficients(coefficient_df):
    return (
        coefficient_df
        .assign(
            absolute_coefficient=lambda df: df["standardized_coefficient"].abs(),
            coefficient_sign=lambda df: np.sign(df["standardized_coefficient"]),
        )
        .groupby("feature", as_index=False)
        .agg(
            mean_abs_standardized_coefficient=("absolute_coefficient", "mean"),
            mean_standardized_coefficient=("standardized_coefficient", "mean"),
            sign_consistency=("coefficient_sign", lambda values: abs(values.mean())),
        )
        .sort_values("mean_abs_standardized_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def summarize_parameter_coefficients(coefficient_df):
    return (
        coefficient_df
        .groupby(["parameter", "feature"], as_index=False)
        .agg(
            mean_coefficient=("standardized_coefficient", "mean"),
            mean_abs_coefficient=("standardized_coefficient", lambda values: values.abs().mean()),
            sign_consistency=("standardized_coefficient", lambda values: abs(np.sign(values).mean())),
        )
        .sort_values(["parameter", "mean_abs_coefficient"], ascending=[True, False])
        .reset_index(drop=True)
    )


def evaluate_ridge_feature_set(
    modeling_df,
    numeric_features,
    categorical_features,
    n_columns,
    accuracy_columns,
):
    all_predictions = []
    all_actual_curves = []

    for held_out_target in sorted(modeling_df["target"].unique()):
        train_df = modeling_df[modeling_df["target"] != held_out_target].copy()
        test_df = modeling_df[modeling_df["target"] == held_out_target].copy()
        y_train = train_df[CURVE_PARAMETER_NAMES].to_numpy(dtype=float)
        feature_columns = categorical_features + numeric_features

        best_alpha = select_ridge_alpha(
            train_df=train_df,
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            feature_columns=feature_columns,
            alpha_values=RIDGE_ALPHAS,
            n_columns=n_columns,
            accuracy_columns=accuracy_columns,
        )

        pipeline = build_ridge_pipeline(numeric_features, categorical_features, best_alpha)
        pipeline.fit(train_df[feature_columns], y_train)
        parameter_predictions = pipeline.predict(test_df[feature_columns])
        curve_predictions = reconstruct_power_law_curves(test_df, parameter_predictions, n_columns)
        actual_curves = test_df[accuracy_columns].to_numpy(dtype=float)

        all_predictions.append(curve_predictions)
        all_actual_curves.append(actual_curves)

    all_predictions = np.vstack(all_predictions)
    all_actual_curves = np.vstack(all_actual_curves)
    point_maes = np.mean(np.abs(all_predictions - all_actual_curves), axis=0)
    return float(np.mean(point_maes))


def leave_one_feature_out_ablation(
    modeling_df,
    numeric_features,
    categorical_features,
    n_columns,
    accuracy_columns,
):
    full_mae = evaluate_ridge_feature_set(
        modeling_df, numeric_features, categorical_features, n_columns, accuracy_columns
    )
    rows = [{"removed_feature": "(none: full model)", "overall_mae": full_mae, "delta_mae_vs_full": 0.0}]

    for feature in numeric_features:
        reduced_numeric_features = [candidate for candidate in numeric_features if candidate != feature]
        reduced_mae = evaluate_ridge_feature_set(
            modeling_df, reduced_numeric_features, categorical_features, n_columns, accuracy_columns
        )
        rows.append({
            "removed_feature": feature,
            "overall_mae": reduced_mae,
            "delta_mae_vs_full": reduced_mae - full_mae,
        })

    if categorical_features:
        reduced_mae = evaluate_ridge_feature_set(
            modeling_df, numeric_features, [], n_columns, accuracy_columns
        )
        rows.append({
            "removed_feature": "model (categorical)",
            "overall_mae": reduced_mae,
            "delta_mae_vs_full": reduced_mae - full_mae,
        })

    return pd.DataFrame(rows).sort_values("delta_mae_vs_full", ascending=False).reset_index(drop=True)


def warm_correlation_outputs(modeling_df):
    correlation_matrix = modeling_df[WARM_NUMERIC_FEATURES].corr()
    pairs = []

    for index, first_feature in enumerate(WARM_NUMERIC_FEATURES):
        for second_feature in WARM_NUMERIC_FEATURES[index + 1:]:
            correlation = float(correlation_matrix.loc[first_feature, second_feature])
            pairs.append({
                "feature_1": first_feature,
                "feature_2": second_feature,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            })

    pair_df = pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)
    return correlation_matrix, pair_df


def grouped_feature_ablation(
    modeling_df,
    numeric_features,
    categorical_features,
    feature_groups,
    n_columns,
    accuracy_columns,
):
    full_mae = evaluate_ridge_feature_set(
        modeling_df, numeric_features, categorical_features, n_columns, accuracy_columns
    )
    rows = [{"removed_group": "(none: full model)", "overall_mae": full_mae, "delta_mae_vs_full": 0.0}]

    for group_name, group_spec in feature_groups.items():
        numeric_to_remove = set(group_spec.get("numeric", []))
        categorical_to_remove = set(group_spec.get("categorical", []))
        reduced_numeric_features = [feature for feature in numeric_features if feature not in numeric_to_remove]
        reduced_categorical_features = [
            feature for feature in categorical_features if feature not in categorical_to_remove
        ]
        reduced_mae = evaluate_ridge_feature_set(
            modeling_df, reduced_numeric_features, reduced_categorical_features, n_columns, accuracy_columns
        )
        rows.append({
            "removed_group": group_name,
            "overall_mae": reduced_mae,
            "delta_mae_vs_full": reduced_mae - full_mae,
        })

    return pd.DataFrame(rows).sort_values("delta_mae_vs_full", ascending=False).reset_index(drop=True)


def main():
    required_files = [
        COLD_START_DATASET_CSV,
        COLD_START_FOLD_SUMMARY_CSV,
        WARM_START_DATASET_CSV,
        WARM_START_FOLD_SUMMARY_CSV,
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Run Stages 4–7 before Stage 8. Missing files:\n" + "\n".join(str(path) for path in missing_files)
        )

    cold_df = pd.read_csv(COLD_START_DATASET_CSV)
    warm_df = pd.read_csv(WARM_START_DATASET_CSV)
    cold_fold_df = pd.read_csv(COLD_START_FOLD_SUMMARY_CSV)
    warm_fold_df = pd.read_csv(WARM_START_FOLD_SUMMARY_CSV)

    require_columns(
        cold_df,
        ["case_id", "source_id", "target"] + CURVE_PARAMETER_NAMES + CATEGORICAL_FEATURES
        + COLD_NUMERIC_FEATURES + COLD_N_COLUMNS + COLD_ACCURACY_COLUMNS,
        "cold-start",
    )
    require_columns(
        warm_df,
        ["case_id", "source_id", "target"] + CURVE_PARAMETER_NAMES + CATEGORICAL_FEATURES
        + WARM_NUMERIC_FEATURES + WARM_N_COLUMNS + WARM_ACCURACY_COLUMNS,
        "warm-start",
    )
    require_columns(cold_fold_df, ["held_out_target", "prediction_model", "selected_alpha"], "cold fold-summary")
    require_columns(warm_fold_df, ["held_out_target", "prediction_model", "selected_alpha"], "warm fold-summary")

    cold_coefficients = ridge_numeric_coefficients_by_fold(
        cold_df, cold_fold_df, "metadata_ridge", COLD_NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    cold_coefficient_summary = summarize_ridge_coefficients(cold_coefficients)
    cold_parameter_coefficients = summarize_parameter_coefficients(cold_coefficients)
    cold_lofo = leave_one_feature_out_ablation(
        cold_df, COLD_NUMERIC_FEATURES, CATEGORICAL_FEATURES, COLD_N_COLUMNS, COLD_ACCURACY_COLUMNS
    )

    warm_coefficients = ridge_numeric_coefficients_by_fold(
        warm_df, warm_fold_df, "metadata_observation_ridge", WARM_NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    warm_coefficient_summary = summarize_ridge_coefficients(warm_coefficients)
    warm_parameter_coefficients = summarize_parameter_coefficients(warm_coefficients)
    warm_correlation_matrix, warm_correlation_pairs = warm_correlation_outputs(warm_df)
    warm_grouped_ablation = grouped_feature_ablation(
        warm_df,
        WARM_NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        WARM_FEATURE_GROUPS,
        WARM_N_COLUMNS,
        WARM_ACCURACY_COLUMNS,
    )

    atomic_save_csv(cold_coefficients, COLD_RIDGE_COEFFICIENTS_CSV)
    atomic_save_csv(cold_coefficient_summary, COLD_RIDGE_COEFFICIENT_SUMMARY_CSV)
    atomic_save_csv(cold_parameter_coefficients, COLD_RIDGE_PARAMETER_COEFFICIENTS_CSV)
    atomic_save_csv(cold_lofo, COLD_LOFO_ABLATION_CSV)
    atomic_save_csv(warm_coefficients, WARM_RIDGE_COEFFICIENTS_CSV)
    atomic_save_csv(warm_coefficient_summary, WARM_RIDGE_COEFFICIENT_SUMMARY_CSV)
    atomic_save_csv(warm_parameter_coefficients, WARM_RIDGE_PARAMETER_COEFFICIENTS_CSV)

    WARM_FEATURE_CORRELATION_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    warm_correlation_matrix.to_csv(WARM_FEATURE_CORRELATION_MATRIX_CSV)
    atomic_save_csv(warm_correlation_pairs, WARM_FEATURE_CORRELATION_PAIRS_CSV)
    atomic_save_csv(warm_grouped_ablation, WARM_GROUPED_ABLATION_CSV)

    print("Saved Stage 8 interpretability outputs to:", COLD_LOFO_ABLATION_CSV.parent)
    print("\nCold-start LOFO ablation:\n", cold_lofo.to_string(index=False))
    print("\nWarm-start grouped ablation:\n", warm_grouped_ablation.to_string(index=False))
    print("\nStrongest warm-start correlations:\n", warm_correlation_pairs.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
