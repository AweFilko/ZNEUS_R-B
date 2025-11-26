import torch
import torch.nn as nn
import torch.optim as optim



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

def train_model(model, train_loader, num_epochs=50, lr=0.001, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #change here for more experiments!!!

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
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    return model, history

