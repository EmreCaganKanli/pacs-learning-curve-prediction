import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import DEVICE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)

        total_loss += loss_function(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return correct / total, total_loss / total


def train_model(model, train_loader, validation_loader, max_epochs = 30, learning_rate = 1e-3,):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,)
    loss_function = nn.CrossEntropyLoss()

    best_state = None
    best_validation_accuracy = -1.0
    best_training_accuracy = -1.0
    best_epoch = 0
    history = []

    progress = tqdm(range(max_epochs), desc = "Current model", leave = True,)

    for epoch_index in progress:
        model.train()

        running_loss = 0.0
        training_correct = 0
        training_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            training_correct += (logits.argmax(1) == labels).sum().item()
            training_total += len(labels)

        training_accuracy = training_correct / training_total
        training_loss = running_loss / len(train_loader)

        validation_accuracy, validation_loss = evaluate(model, validation_loader,)

        epoch_number = epoch_index + 1

        history.append({
            "epoch": epoch_number,
            "train_accuracy": training_accuracy,
            "val_accuracy": validation_accuracy,
            "train_loss": training_loss,
            "val_loss": validation_loss,
        })

        progress.set_description(
            f"Epoch {epoch_number}/{max_epochs}"
        )

        progress.set_postfix(
            train_loss=f"{training_loss:.3f}",
            train_acc=f"{training_accuracy:.3f}",
            val_acc=f"{validation_accuracy:.3f}",
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_training_accuracy = training_accuracy
            best_epoch = epoch_number
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    training_info = {
        "epochs_completed": max_epochs,
        "best_epoch": best_epoch,
        "best_train_accuracy": best_training_accuracy,
        "best_val_accuracy": best_validation_accuracy,
        "optimization_steps": max_epochs * len(train_loader),
        "history": history,
    }

    return model, training_info


def save_training_plot(
    history_df,
    best_epoch,
    best_validation_accuracy,
    title,
    output_path,
    training_label="Training accuracy",
    validation_label="Validation accuracy",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(history_df["epoch"], history_df["train_accuracy"], label = training_label,)

    plt.plot(history_df["epoch"], history_df["val_accuracy"], label = validation_label,)

    plt.scatter(
        best_epoch,
        best_validation_accuracy,
        s=80,
        label=(
            f"Best validation: {best_validation_accuracy:.3f} "
            f"at epoch {best_epoch}"
        ),
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def atomic_save_csv(dataframe, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_name(
        f"{destination.stem}_temp{destination.suffix}"
    )

    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)