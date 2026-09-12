import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (ADAPTATION_CSV, CURVE_TARGETS_CSV, POWER_LAW_C_MAX, POWER_LAW_C_MIN,)
from prediction import power_law_curve
from training import atomic_save_csv


def fit_case_curve(case_df):
    case_df = case_df.sort_values("target_train_size")

    a0 = float(case_df["zero_shot_test_accuracy"].iloc[0])

    n_values = np.concatenate([[0.0], case_df["target_train_size"].to_numpy(dtype = float),])

    accuracy_values = np.concatenate([
        [a0],
        case_df["adapted_test_accuracy"].to_numpy(dtype=float),
    ])

    def curve_with_fixed_a0(n, a, c):
        return power_law_curve(n, a0, a, c,)

    initial_a = min(1.0, max(a0 + 0.01, accuracy_values.max(),),)

    if initial_a >= 1.0:
        initial_a = 0.999999

    parameters, _ = curve_fit(
        curve_with_fixed_a0,
        n_values,
        accuracy_values,
        p0=[initial_a, 0.3],
        bounds=([a0, POWER_LAW_C_MIN], [1.0, POWER_LAW_C_MAX],),
        maxfev=20000,
    )

    curve_a, curve_c = parameters

    fitted_accuracies = power_law_curve(n_values, a0, curve_a, curve_c,)

    errors = fitted_accuracies - accuracy_values

    return {
        "curve_a0": a0,
        "curve_a": float(curve_a),
        "curve_c": float(curve_c),
        "curve_fit_mae": float(np.mean(np.abs(errors))),
        "curve_fit_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "curve_fit_max_error": float(np.max(np.abs(errors))),
    }


def main():
    if not ADAPTATION_CSV.exists():
        raise FileNotFoundError("Run stages/02_run_adaptation.py first.")

    adaptation_df = pd.read_csv(ADAPTATION_CSV)

    required_columns = [
        "case_id",
        "zero_shot_val_accuracy",
        "zero_shot_val_loss",
        "zero_shot_test_accuracy",
        "zero_shot_test_loss",
        "target_train_size",
        "adapted_test_accuracy",
    ]

    missing_columns = [column for column in required_columns if column not in adaptation_df.columns]

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    case_metadata = (
        adaptation_df[
            [
                "case_id",
                "source_id",
                "model",
                "sources",
                "source_count",
                "target",
                "seed",
                "parameter_count",
                "source_size",
                "source_accuracy",
                "zero_shot_val_accuracy",
                "zero_shot_val_loss",
                "zero_shot_test_accuracy",
                "zero_shot_test_loss",
            ]
        ]
        .drop_duplicates("case_id")
        .set_index("case_id")
    )

    accuracy_wide = (
        adaptation_df.pivot(
            index="case_id",
            columns="fraction_percent",
            values="adapted_test_accuracy",
        )
        .rename(
            columns=lambda value: (
                f"accuracy_{int(value)}"
            )
        )
    )

    n_wide = (
        adaptation_df.pivot(
            index="case_id",
            columns="fraction_percent",
            values="target_train_size",
        )
        .rename(
            columns=lambda value: (
                f"n_{int(value)}"
            )
        )
    )

    curve_df = (case_metadata.join(n_wide).join(accuracy_wide).reset_index())

    curve_df["n_0"] = 0
    curve_df["accuracy_0"] = (curve_df["zero_shot_test_accuracy"])

    n_columns = ["n_0", "n_5", "n_10", "n_25", "n_50", "n_100",]

    accuracy_columns = [
        "accuracy_0",
        "accuracy_5",
        "accuracy_10",
        "accuracy_25",
        "accuracy_50",
        "accuracy_100",
    ]

    required_curve_columns = (n_columns + accuracy_columns)

    missing_columns = [
        column
        for column in required_curve_columns
        if column not in curve_df.columns
    ]

    if missing_columns:
        raise ValueError("Missing curve values: " + ", ".join(missing_columns))

    if curve_df[required_curve_columns].isna().any().any():
        raise ValueError("Some cases do not have all " "required curve points.")

    if not curve_df["case_id"].is_unique:
        raise ValueError("curve_targets.csv must contain " "one row per case_id.")

    fitted_rows = []

    for case_id, case_df in adaptation_df.groupby("case_id"):
        curve_parameters = fit_case_curve(case_df)

        fitted_rows.append({"case_id": case_id, ** curve_parameters,})

    parameters_df = pd.DataFrame(fitted_rows)

    curve_df = curve_df.merge(parameters_df, on = "case_id", how = "left", validate = "one_to_one",)

    atomic_save_csv(curve_df, CURVE_TARGETS_CSV,)

    print("Saved curve targets to:", CURVE_TARGETS_CSV,)

    print("Curve-target shape:", curve_df.shape,)

    print("Mean fitted-curve MAE:", curve_df["curve_fit_mae"].mean(),)

    print("Maximum fitted-curve MAE:", curve_df["curve_fit_mae"].max(),)


if __name__ == "__main__":
    main()