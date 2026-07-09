"""
trainer.py
----------
Training utilities for image classification models.
"""

import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (
    correct_predictions / total_samples
) * 100

    return epoch_loss, epoch_accuracy
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate the model for one epoch.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
):
    """
    Train the model for multiple epochs.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):

        print(f"\n{'=' * 50}")
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"{'=' * 50}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_accuracy:.2f}%"
)

        print(
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_accuracy:.2f}%"
)

        print(
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_accuracy:.2f}%"
)

    return history