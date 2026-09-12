import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ADAPTATION_CSV,
    ADAPTATION_EPOCHS,
    ADAPTATION_HISTORY_DIR,
    ADAPTATION_LR,
    ADAPTATION_MODEL_DIR,
    ADAPTATION_PLOT_DIR,
    DEVICE,
    FRACTIONS,
    SOURCE_CSV,
)
from data import build_data_state, make_loader
from models import build_model
from training import (atomic_save_csv, evaluate, save_training_plot, set_seed, train_model,)


def build_adaptation_cases(source_df, domains):
    cases = []

    for _, row in source_df.iterrows():
        sources = tuple(str(row["sources"]).split("|"))

        for target in domains:
            if target in sources:
                continue

            cases.append({
                "case_id": (
                    f"{row['source_id']}"
                    f"_to_{target}"
                ),
                "source_id": row["source_id"],
                "model": row["model"],
                "sources": sources,
                "target": target,
                "seed": row["seed"],
                "checkpoint": row["checkpoint"],
                "parameter_count": (row["parameter_count"]),
                "source_size": row["source_size"],
                "source_accuracy": (row["source_accuracy"]),
            })

    return pd.DataFrame(cases)


def load_valid_adaptation_results():
    if not ADAPTATION_CSV.exists():
        return pd.DataFrame()

    adaptation_df = pd.read_csv(ADAPTATION_CSV)

    valid_rows = []

    for _, row in adaptation_df.iterrows():
        checkpoint_exists = (
            pd.notna(row.get("adapted_checkpoint"))
            and Path(row["adapted_checkpoint"]).exists()
        )

        history_exists = (pd.notna(row.get("history_csv")) and Path(row["history_csv"]).exists())

        if checkpoint_exists and history_exists:
            valid_rows.append(row.to_dict())

    valid_df = pd.DataFrame(valid_rows)

    if not valid_df.empty:
        if "zero_shot_accuracy" in valid_df.columns:
            if "zero_shot_test_accuracy" in valid_df.columns:
                valid_df = valid_df.drop(columns = ["zero_shot_accuracy"])
            else:
                valid_df = valid_df.rename(
                    columns={"zero_shot_accuracy": "zero_shot_test_accuracy"}
                )

        if "zero_shot_loss" in valid_df.columns:
            if "zero_shot_test_loss" in valid_df.columns:
                valid_df = valid_df.drop(columns = ["zero_shot_loss"])
            else:
                valid_df = valid_df.rename(columns = {"zero_shot_loss": "zero_shot_test_loss"})

    atomic_save_csv(valid_df, ADAPTATION_CSV,)

    return valid_df


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError("Run stages/01_train_sources.py first.")

    data_state = build_data_state()
    domains = data_state["domains"]
    splits = data_state["splits"]
    target_subsets = data_state["target_subsets"]

    source_df = pd.read_csv(SOURCE_CSV)
    cases_df = build_adaptation_cases(source_df, domains,)

    adaptation_df = load_valid_adaptation_results()

    completed_run_ids = (
        set(adaptation_df["run_id"].astype(str))
        if not adaptation_df.empty
        else set()
    )

    total_runs = (len(cases_df) * len(FRACTIONS))

    print(
        f"Total adaptation runs: "
        f"{total_runs}"
    )

    print(
        f"Already completed: "
        f"{len(completed_run_ids)}"
    )

    for case_number, case in cases_df.iterrows():
        case_id = str(case["case_id"])
        source_id = str(case["source_id"])
        model_name = str(case["model"])
        target = str(case["target"])
        source_seed = int(case["seed"])

        expected_run_ids = {
            (
                f"{case_id}_fraction"
                f"{int(round(fraction * 100))}"
            )
            for fraction in FRACTIONS
        }

        case_runs_complete = (expected_run_ids.issubset(completed_run_ids))

        if adaptation_df.empty:
            case_mask = pd.Series(False, index = adaptation_df.index, dtype = bool,)
        else:
            case_mask = (adaptation_df["case_id"].astype(str) == case_id)

        case_has_zero_shot_validation = (
            case_mask.any()
            and ("zero_shot_val_accuracy" in adaptation_df.columns)
            and ("zero_shot_val_loss" in adaptation_df.columns)
            and adaptation_df.loc[case_mask, "zero_shot_val_accuracy",].notna().all()
            and adaptation_df.loc[case_mask, "zero_shot_val_loss",].notna().all()
        )

        case_has_zero_shot_test = (
            case_mask.any()
            and ("zero_shot_test_accuracy" in adaptation_df.columns)
            and ("zero_shot_test_loss" in adaptation_df.columns)
            and adaptation_df.loc[case_mask, "zero_shot_test_accuracy",].notna().all()
            and adaptation_df.loc[case_mask, "zero_shot_test_loss",].notna().all()
        )

        if (case_runs_complete and case_has_zero_shot_validation and case_has_zero_shot_test):
            print(
                f"Skipping completed case "
                f"{case_number + 1}/"
                f"{len(cases_df)}: "
                f"{source_id} -> {target}"
            )
            continue

        source_state = torch.load(
            Path(case["checkpoint"]),
            map_location="cpu",
            weights_only=True,
        )

        target_validation_loader = make_loader(splits[target]["val"], train = False,)

        target_test_loader = make_loader(splits[target]["test"], train = False,)

        zero_shot_model = build_model(model_name)

        zero_shot_model.load_state_dict(source_state)

        zero_shot_model = zero_shot_model.to(DEVICE)

        (zero_shot_val_accuracy, zero_shot_val_loss,) = evaluate(
            zero_shot_model,
            target_validation_loader,
        )

        if case_has_zero_shot_test:
            zero_shot_test_accuracy = float(
                adaptation_df.loc[case_mask, "zero_shot_test_accuracy",].iloc[0]
            )

            zero_shot_test_loss = float(
                adaptation_df.loc[case_mask, "zero_shot_test_loss",].iloc[0]
            )

        else:
            (zero_shot_test_accuracy, zero_shot_test_loss,) = evaluate(
                zero_shot_model,
                target_test_loader,
            )

        if case_mask.any():
            adaptation_df.loc[case_mask, "zero_shot_val_accuracy",] = zero_shot_val_accuracy

            adaptation_df.loc[case_mask, "zero_shot_val_loss",] = zero_shot_val_loss

            adaptation_df.loc[case_mask, "zero_shot_test_accuracy",] = zero_shot_test_accuracy

            adaptation_df.loc[case_mask, "zero_shot_test_loss",] = zero_shot_test_loss

            atomic_save_csv(adaptation_df, ADAPTATION_CSV,)

        del zero_shot_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if case_runs_complete:
            print(
                "Added missing zero-shot metrics: "
                f"{source_id} -> {target}"
            )
            continue

        for fraction in FRACTIONS:
            fraction_percent = int(round(fraction * 100))

            run_id = (
                f"{case_id}_fraction"
                f"{fraction_percent}"
            )

            if run_id in completed_run_ids:
                print(
                    f"Skipping completed run: "
                    f"{run_id}"
                )
                continue

            print(
                f"\nFine-tuning "
                f"target={target}, "
                f"fraction={fraction:.2f}"
            )

            set_seed(source_seed + fraction_percent)

            target_train_indices = (target_subsets[target][fraction])

            target_train_loader = make_loader(target_train_indices, train = True,)

            adaptation_model = build_model(model_name)

            adaptation_model.load_state_dict(source_state)

            (adaptation_model, training_info,) = train_model(
                adaptation_model,
                target_train_loader,
                target_validation_loader,
                max_epochs=ADAPTATION_EPOCHS,
                learning_rate=ADAPTATION_LR,
            )

            (target_test_accuracy, target_test_loss,) = evaluate(
                adaptation_model,
                target_test_loader,
            )

            adapted_checkpoint_path = (
                ADAPTATION_MODEL_DIR
                / f"{run_id}.pth"
            )

            torch.save(adaptation_model.state_dict(), adapted_checkpoint_path,)

            history_path = (
                ADAPTATION_HISTORY_DIR
                / f"{run_id}_history.csv"
            )

            plot_path = (
                ADAPTATION_PLOT_DIR
                / f"{run_id}.png"
            )

            history_df = pd.DataFrame(training_info["history"])

            history_df.to_csv(history_path, index = False,)

            save_training_plot(
                history_df=history_df,
                best_epoch=training_info["best_epoch"],
                best_validation_accuracy=(training_info["best_val_accuracy"]),
                title=(
                    f"{source_id} -> "
                    f"{target}, "
                    f"fraction={fraction:.2f}"
                ),
                output_path=plot_path,
                training_label=("Target training accuracy"),
                validation_label=("Target validation accuracy"),
            )

            new_row = {
                "run_id": run_id,
                "case_id": case_id,
                "source_id": source_id,
                "model": model_name,
                "sources": "|".join(case["sources"]),
                "source_count": len(case["sources"]),
                "target": target,
                "seed": source_seed,
                "fraction": fraction,
                "fraction_percent": (fraction_percent),
                "parameter_count": (case["parameter_count"]),
                "source_size": (case["source_size"]),
                "source_accuracy": (case["source_accuracy"]),
                "target_train_size": len(target_train_indices),
                "target_val_size": len(splits[target]["val"]),
                "target_test_size": len(splits[target]["test"]),
                "zero_shot_val_accuracy": (zero_shot_val_accuracy),
                "zero_shot_val_loss": (zero_shot_val_loss),
                "zero_shot_test_accuracy": (zero_shot_test_accuracy),
                "zero_shot_test_loss": (zero_shot_test_loss),
                "adapted_test_accuracy": (target_test_accuracy),
                "adapted_test_loss": (target_test_loss),
                "accuracy_gain": (target_test_accuracy - zero_shot_test_accuracy),
                "best_epoch": (training_info["best_epoch"]),
                "best_train_accuracy": (training_info["best_train_accuracy"]),
                "best_val_accuracy": (training_info["best_val_accuracy"]),
                "epochs_completed": (training_info["epochs_completed"]),
                "optimization_steps": (training_info["optimization_steps"]),
                "adaptation_epochs": (ADAPTATION_EPOCHS),
                "adaptation_learning_rate": (ADAPTATION_LR),
                "adapted_checkpoint": str(adapted_checkpoint_path),
                "history_csv": str(history_path),
                "plot": str(plot_path),
            }

            if not adaptation_df.empty:
                adaptation_df = adaptation_df[adaptation_df["run_id"] != run_id]

            adaptation_df = pd.concat(
                [adaptation_df, pd.DataFrame([new_row]),],
                ignore_index=True,
            )

            atomic_save_csv(adaptation_df, ADAPTATION_CSV,)

            completed_run_ids.add(run_id)

            del adaptation_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_df = (pd.read_csv(ADAPTATION_CSV) if ADAPTATION_CSV.exists() else pd.DataFrame())

    print(
        "\nCompleted adaptation runs: "
        f"{len(final_df)}/{total_runs}"
    )


if __name__ == "__main__":
    main()