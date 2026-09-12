import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    MAX_SOURCE_EPOCHS,
    MODEL_NAMES,
    SEEDS,
    SOURCE_CSV,
    SOURCE_HISTORY_DIR,
    SOURCE_LR,
    SOURCE_MODEL_DIR,
    SOURCE_PLOT_DIR,
)
from data import build_data_state, make_loader
from models import build_model
from training import (atomic_save_csv, evaluate, save_training_plot, set_seed, train_model,)


def valid_checkpoint(path, model_name):
    path = Path(path)

    if not path.exists():
        return False

    try:
        state_dict = torch.load(path, map_location = "cpu", weights_only = True,)

        model = build_model(model_name)
        model.load_state_dict(state_dict)

        return True

    except Exception:
        return False


def load_valid_source_results():
    if not SOURCE_CSV.exists():
        return pd.DataFrame()

    source_df = pd.read_csv(SOURCE_CSV)
    valid_rows = []

    for _, row in source_df.iterrows():
        checkpoint_path = Path(row["checkpoint"])
        history_path = Path(row["history_csv"])
        plot_path = Path(row["plot"])

        if (
            valid_checkpoint(checkpoint_path, row["model"],)
            and history_path.exists()
            and plot_path.exists()
        ):
            valid_rows.append(row.to_dict())

    valid_df = pd.DataFrame(valid_rows)
    atomic_save_csv(valid_df, SOURCE_CSV)

    return valid_df


def main():
    data_state = build_data_state()
    domains = data_state["domains"]
    splits = data_state["splits"]

    source_configurations = []

    for source_count in [1, 2]:
        source_configurations.extend(itertools.combinations(domains, source_count,))

    source_df = load_valid_source_results()

    complete_source_ids = (
        set(source_df["source_id"].astype(str))
        if not source_df.empty
        else set()
    )

    all_source_cases = []

    for model_name in MODEL_NAMES:
        for sources in source_configurations:
            for seed in SEEDS:
                source_id = (
                    f"{model_name}_"
                    f"{'-'.join(sources)}_"
                    f"seed{seed}"
                )

                checkpoint_path = (
                    SOURCE_MODEL_DIR
                    / f"{source_id}.pth"
                )

                history_path = (
                    SOURCE_HISTORY_DIR
                    / f"{source_id}_history.csv"
                )

                plot_path = (
                    SOURCE_PLOT_DIR
                    / f"{source_id}.png"
                )

                complete = (
                    source_id in complete_source_ids
                    and valid_checkpoint(checkpoint_path, model_name,)
                    and history_path.exists()
                    and plot_path.exists()
                )

                if not complete:
                    all_source_cases.append((model_name, sources, seed))

    print("Remaining source trainings:", len(all_source_cases),)

    for case_number, (model_name, sources, seed,) in enumerate(all_source_cases, start=1):
        source_id = (
            f"{model_name}_"
            f"{'-'.join(sources)}_"
            f"seed{seed}"
        )

        checkpoint_path = (
            SOURCE_MODEL_DIR / f"{source_id}.pth"
        )

        history_path = (
            SOURCE_HISTORY_DIR
            / f"{source_id}_history.csv"
        )

        plot_path = (
            SOURCE_PLOT_DIR / f"{source_id}.png"
        )

        print(
            f"\nSource model "
            f"{case_number}/{len(all_source_cases)}: "
            f"{model_name}, "
            f"sources={sources}, "
            f"seed={seed}"
        )

        set_seed(seed)

        source_train_indices = np.concatenate([splits[domain]["train"] for domain in sources])

        source_validation_indices = np.concatenate([splits[domain]["val"] for domain in sources])

        train_loader = make_loader(source_train_indices, train = True,)

        validation_loader = make_loader(source_validation_indices, train = False,)

        model = build_model(model_name)

        model, training_info = train_model(
            model,
            train_loader,
            validation_loader,
            max_epochs=MAX_SOURCE_EPOCHS,
            learning_rate=SOURCE_LR,
        )

        source_accuracy, source_loss = evaluate(model, validation_loader,)

        torch.save(model.state_dict(), checkpoint_path,)

        history_df = pd.DataFrame(training_info["history"])

        history_df.to_csv(history_path, index = False,)

        save_training_plot(
            history_df=history_df,
            best_epoch=training_info["best_epoch"],
            best_validation_accuracy=(training_info["best_val_accuracy"]),
            title=source_id,
            output_path=plot_path,
        )

        new_row = {
            "source_id": source_id,
            "model": model_name,
            "sources": "|".join(sources),
            "source_count": len(sources),
            "seed": seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "source_size": len(source_train_indices),
            "source_val_size": len(source_validation_indices),
            "source_accuracy": source_accuracy,
            "source_loss": source_loss,
            "max_epochs": MAX_SOURCE_EPOCHS,
            "epochs_completed": (training_info["epochs_completed"]),
            "best_epoch": (training_info["best_epoch"]),
            "best_train_accuracy": (training_info["best_train_accuracy"]),
            "best_val_accuracy": (training_info["best_val_accuracy"]),
            "optimization_steps": (training_info["optimization_steps"]),
            "learning_rate": SOURCE_LR,
            "batch_size": BATCH_SIZE,
            "checkpoint": str(checkpoint_path),
            "history_csv": str(history_path),
            "plot": str(plot_path),
        }

        if not source_df.empty:
            source_df = source_df[source_df["source_id"] != source_id]

        source_df = pd.concat([source_df, pd.DataFrame([new_row]),], ignore_index = True,)

        atomic_save_csv(source_df, SOURCE_CSV,)

        print(
            f"Saved completed model: "
            f"{source_id}"
        )

    final_df = (pd.read_csv(SOURCE_CSV) if SOURCE_CSV.exists() else pd.DataFrame())

    print("\nCompleted source models:", len(final_df),)


if __name__ == "__main__":
    main()