import torch.optim as optim
import wandb
from model import *

# SimpleCNN:
# # LR = 0.001
# # Epochs = 20–35
# # Batch size = 8–16
# # Dropout = 0.3
# # Weight decay = 1e-4
#
# DeepCNN:
# # LR = 0.0005
# # Epochs = 25–40
# # Batch size = 8
# # Dropout = 0.4–0.5
# # Weight decay = 5e-4

def complex_train(model, train_loader, val_loader, cfg=None):

    model, criterion, optimizer, scheduler, num_epochs, device = setup(cfg, model)

    history = {"loss": [], "acc": []}

    for epoch in range(num_epochs):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        model.eval()
        val_total = 0
        val_correct = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * imgs.size(0)
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{num_epochs}  "
              f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step(val_loss)

    return model, history


def simple_train(model, train_loader, cfg=None):

    model, criterion, optimizer, scheduler, num_epochs, device = setup(cfg, model)

    history = {"loss": [], "acc": []}

    for epoch in range(num_epochs):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        model.eval()


        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{num_epochs}  "
              f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        if scheduler:
            scheduler.step(epoch_loss)

    return model, history


def setup(cfg, model):
    num_epochs = int(cfg['model_hyperparams']["epochs"])
    lr = float(cfg['model_hyperparams']['learning_rate'])
    device = cfg['setup']['device']
    weight_decay = float(cfg['model_hyperparams']['weight_decay'])
    mode = cfg['model_hyperparams']['mode']
    factor = float(cfg['model_hyperparams']['factor'])
    patience = int(cfg['model_hyperparams']['patience'])

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # change here for more experiments!!!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
    )

    return model, criterion, optimizer, scheduler, num_epochs, device
