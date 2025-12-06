import torch.optim as optim
import wandb
from model import *

def complex_train(model, train_loader, val_loader, cfg=None):

    model, criterion, optimizer, scheduler, num_epochs, device = setup(cfg, model)

    history = {"loss": [], "acc": []}

    # best_acc = 0
    # name = wandb.run.name

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
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), f"{name}_model.pt")

        print(f"Epoch {epoch + 1}/{num_epochs}  "
              f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if scheduler:
            if int(cfg['scheduler']['choice'])== 0:
                scheduler.step(epoch_loss)
            elif int(cfg['scheduler']['choice'])==1:
                scheduler.step()

    # state = torch.load(f"{name}_model.pt",weights_only=True)
    # model.load_state_dict(state)

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
            if int(cfg['scheduler']['choice'])== 0:
                scheduler.step(epoch_loss)
            elif int(cfg['scheduler']['choice'])==1:
                scheduler.step()

    return model, history


def setup(cfg, model):
    num_epochs = int(cfg['model_hyperparams']["epochs"])
    device = cfg['setup']['device']
    lr = float(cfg['optimizer']['learning_rate'])
    weight_decay = float(cfg['optimizer']['weight_decay'])
    optimizer = None

    if int(cfg['optimizer']['choice']) == 0:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif int(cfg['optimizer']['choice']) == 1:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if int(cfg['scheduler']['choice']) == 0:
        mode = cfg['scheduler']['mode']
        factor = float(cfg['scheduler']['factor'])
        patience = int(cfg['scheduler']['patience'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    elif int(cfg['scheduler']['choice']) == 1:
        t_max = int(cfg['scheduler']['t_max'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        scheduler = None

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    return model, criterion, optimizer, scheduler, num_epochs, device
