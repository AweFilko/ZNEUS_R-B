import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from model import *

def get_metrics(model, loader, device=None):

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return acc, prec, rec, f1, np.array(all_labels), np.array(all_preds)


def plot_training_curves(history, title_prefix="Model"):
    loss = history["loss"]
    acc = history["acc"]

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss)
    plt.title(f"{title_prefix} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc)
    plt.title(f"{title_prefix} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def confused_mat(y_true, y_pred, class_names):
    #hehe
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def compare_models(results_dict):

    print("\n=== Model Comparison ===")
    print(f"{'Model':12}  ACC     PREC     REC      F1")
    print("-" * 46)

    for name, (acc, prec, rec, f1) in results_dict.items():
        print(f"{name:12}  {acc:.4f}  {prec:.4f}  {rec:.4f}  {f1:.4f}")

def evaluate_model(model, loader, device, cfg):

    if isinstance(model, SCNN):
        print("\n=== SimpleCNN Analysis ===")
    elif isinstance(model, DCNN):
        print("\n=== DeepCNN Analysis ===")
    else:
        print("\n=== FENN Analysis ===")

    acc, prec, rec, f1, y_true, y_pred = get_metrics(model, loader, device)
    print(f"ACC: {acc}| PREC: {prec}| REC: {rec}| F1: {f1}")

    wandb.log({"accuracy": acc,
               "precision": prec,
               "recall": rec,
               "F1": f1})

    confused_mat(y_true, y_pred, cfg['setup']['sport'])
