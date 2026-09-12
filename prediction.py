import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import POWER_LAW_C_MAX, POWER_LAW_C_MIN


def power_law_curve(n, a0, a, c):
    n = np.asarray(n, dtype=float)

    return (a0 + (a - a0) * (1.0 - np.power(n + 1.0, -c)))


def sanitize_curve_parameters(parameters):
    parameters = np.asarray(parameters, dtype=float).copy()

    parameters[:, 0] = np.clip(parameters[:, 0], 0.0, 1.0,)

    parameters[:, 1] = np.clip(parameters[:, 1], 0.0, 1.0,)

    parameters[:, 1] = np.maximum(parameters[:, 1], parameters[:, 0],)

    parameters[:, 2] = np.clip(parameters[:, 2], POWER_LAW_C_MIN, POWER_LAW_C_MAX,)

    return parameters


def reconstruct_power_law_curves(dataframe, parameter_predictions, n_columns,):
    parameter_predictions = sanitize_curve_parameters(parameter_predictions)

    predicted_curves = []

    for row_index in range(len(dataframe)):
        row = dataframe.iloc[row_index]
        a0, a, c = parameter_predictions[row_index]

        n_values = np.array([row[column] for column in n_columns], dtype = float,)

        predictions = power_law_curve(n_values, a0, a, c,)

        predicted_curves.append(np.clip(predictions, 0.0, 1.0))

    return np.asarray(predicted_curves, dtype = float,)


def build_ridge_pipeline(numeric_features, categorical_features, alpha,):
    transformers = []

    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features,))

    if categorical_features:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown = "ignore",), categorical_features,)
        )

    preprocessor = ColumnTransformer(transformers = transformers, remainder = "drop",)

    return Pipeline([("preprocessor", preprocessor), ("ridge", Ridge(alpha = alpha)),])


def select_ridge_alpha(
    train_df,
    y_train,
    numeric_features,
    categorical_features,
    feature_columns,
    alpha_values,
    n_columns,
    accuracy_columns,
):
    inner_groups = (train_df["source_id"].astype(str).to_numpy())

    unique_group_count = len(np.unique(inner_groups))

    inner_split_count = min(4, unique_group_count,)

    if inner_split_count < 2:
        raise ValueError("Not enough source_id groups for " "inner cross-validation.")

    inner_cv = GroupKFold(n_splits = inner_split_count)

    alpha_results = []

    for alpha in alpha_values:
        inner_fold_maes = []

        for (inner_train_indices, inner_validation_indices,) in inner_cv.split(
            train_df,
            y_train,
            groups=inner_groups,
        ):
            inner_train_df = train_df.iloc[inner_train_indices]

            inner_validation_df = train_df.iloc[inner_validation_indices]

            inner_y_train = y_train[inner_train_indices]

            pipeline = build_ridge_pipeline(numeric_features, categorical_features, alpha,)

            pipeline.fit(inner_train_df[feature_columns], inner_y_train,)

            parameter_predictions = pipeline.predict(inner_validation_df[feature_columns])

            predicted_curves = reconstruct_power_law_curves(
                inner_validation_df,
                parameter_predictions,
                n_columns,
            )

            actual_curves = (inner_validation_df[accuracy_columns].to_numpy(dtype = float))

            inner_fold_maes.append(mean_absolute_error(actual_curves, predicted_curves,))

        alpha_results.append({
            "alpha": float(alpha),
            "mean_inner_mae": float(np.mean(inner_fold_maes)),
        })

    results_df = (
        pd.DataFrame(alpha_results)
        .sort_values(["mean_inner_mae", "alpha"])
        .reset_index(drop=True)
    )

    return float(results_df.iloc[0]["alpha"])


def three_point_log_curve_predictions(
    zero_shot_accuracy,
    first_n,
    first_accuracy,
    second_n,
    second_accuracy,
    prediction_n,
):
    zero_shot_accuracy = np.asarray(zero_shot_accuracy, dtype = float,)

    first_n = np.asarray(first_n, dtype = float,)

    first_accuracy = np.asarray(first_accuracy, dtype = float,)

    second_n = np.asarray(second_n, dtype = float,)

    second_accuracy = np.asarray(second_accuracy, dtype = float,)

    prediction_n = np.asarray(prediction_n, dtype = float,)

    all_predictions = []

    for row_index in range(len(zero_shot_accuracy)):
        observed_n = np.array([0.0, first_n[row_index], second_n[row_index],], dtype = float,)

        observed_accuracy = np.array(
            [zero_shot_accuracy[row_index], first_accuracy[row_index], second_accuracy[row_index],],
            dtype=float,
        )

        observed_design = np.column_stack([np.ones(3), np.log1p(observed_n),])

        coefficients = np.linalg.lstsq(observed_design, observed_accuracy, rcond = None,)[0]

        row_prediction_n = prediction_n[row_index]

        prediction_design = np.column_stack([
            np.ones(len(row_prediction_n)),
            np.log1p(row_prediction_n),
        ])

        all_predictions.append(prediction_design @ coefficients)

    return np.clip(np.asarray(all_predictions), 0.0, 1.0,)