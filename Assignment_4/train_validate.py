from data_process import get_data
import models

import torch
import torch.nn as nn
import torch.optim as optim

# check if the device use GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get data
train_dataset, val_dataset, test_dataset = get_data()

# get loader
# train loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# initialize model
# baseline
model = models.LeNet5_baseline().to(device)

# loss func and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train & test
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")
