import copy
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE, EPOCHS, LR, FINETUNE_EPOCHS, FINETUNE_LR,
    PRUNE_RATIOS, NUM_CLASSES, RESULTS_CSV,
    PRUNING_PLOT, TRAINING_PLOT
)
from src.model import TrafficCNN
from src.prune_utils import apply_global_pruning, measure_sparsity

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history

def fine_tune(model, train_loader, val_loader, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, val_acc = evaluate(model, val_loader)
        print(f"Fine-tune Epoch [{epoch+1}/{epochs}] Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_acc

def run_pruning_experiments(model, train_loader, val_loader, test_loader):
    baseline_state = copy.deepcopy(model.state_dict())
    results = []

    for ratio in PRUNE_RATIOS:
        print(f"\nTesting pruning ratio: {ratio}")

        pruned_model = TrafficCNN(num_classes=NUM_CLASSES).to(DEVICE)
        pruned_model.load_state_dict(baseline_state)

        pruned_model = apply_global_pruning(pruned_model, amount=ratio)

        _, val_acc_before = evaluate(pruned_model, val_loader)
        _, test_acc_before = evaluate(pruned_model, test_loader)

        pruned_model, val_acc_after = fine_tune(
            pruned_model,
            train_loader,
            val_loader
        )

        _, test_acc_after = evaluate(pruned_model, test_loader)
        sparsity = measure_sparsity(pruned_model)

        results.append({
            "pruning_ratio": ratio,
            "val_acc_before": val_acc_before,
            "test_acc_before": test_acc_before,
            "val_acc_after": val_acc_after,
            "test_acc_after": test_acc_after,
            "sparsity": sparsity
        })

        print(
            f"Ratio: {ratio:.2f} | "
            f"Val before: {val_acc_before:.4f} | "
            f"Test before: {test_acc_before:.4f} | "
            f"Val after: {val_acc_after:.4f} | "
            f"Test after: {test_acc_after:.4f} | "
            f"Sparsity: {sparsity:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    return results_df

def save_pruning_plot(results_df):
    plt.figure(figsize=(9, 5))
    plt.plot(results_df["pruning_ratio"], results_df["val_acc_before"], marker="o", label="Val Acc Before FT")
    plt.plot(results_df["pruning_ratio"], results_df["test_acc_before"], marker="o", label="Test Acc Before FT")
    plt.plot(results_df["pruning_ratio"], results_df["val_acc_after"], marker="s", label="Val Acc After FT")
    plt.plot(results_df["pruning_ratio"], results_df["test_acc_after"], marker="^", label="Test Acc After FT")
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Pruning Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRUNING_PLOT)
    plt.close()

def save_training_plot(history):
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline Training Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_PLOT)
    plt.close()
