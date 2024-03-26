from data_process import get_data
import models

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, device, optimizer, loss_func):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # calculate LOSS & ACC
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    return model, train_loss, train_acc

def validate(model, val_loader, device, loss_func):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            outputs = model(images)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # calculate LOSS & ACC
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total

    return model, val_loss, val_acc

def get_dir():
    # find the next available subdirectory name
    save_dir = 'models'
    sub_dirs = os.listdir(save_dir)
    sub_dirs = [name for name in sub_dirs if os.path.isdir(os.path.join(save_dir, name))]
    sub_dirs = sorted(sub_dirs, key=lambda x: int(x))
    if len(sub_dirs) > 0:
        new_dir_num = 1 + int(sub_dirs[-1])
    else:
        new_dir_num = 1

    # create the subdirectory for saving the trained model
    sub_dir = f'{new_dir_num:02}'
    save_path = os.path.join(save_dir, sub_dir)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def save_model(model, save_path):
    # save the trained model in the subdirectory
    save_file = os.path.join(save_path, 'saved_model.pth')
    torch.save(model.state_dict(), save_file)
    print(f"Model saved in {save_file}.")

def save_log(epoch, train_loss, train_acc, val_loss, val_acc, save_path, model_name):
    log_path = os.path.join(save_path, 'training.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Training model: {model_name}')
    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

def train_validate(num_epochs):
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # get data
    train_dataset, val_dataset, _ = get_data()
    # get loader
    # train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # validate loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # initialize model
    # baseline
    model = models.LeNet5_baseline().to(device)
    model_name = model.__class__.__name__

    # Model save path
    save_path = get_dir()

    # loss func and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train & test

    for epoch in range(num_epochs):

        model, train_loss, train_acc = train(model, train_loader, device, optimizer, loss_func)
        model, val_loss, val_acc = validate(model, val_loader, device, loss_func)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        save_log(epoch, train_loss, train_acc, val_loss, val_acc, save_path, model_name)

    # Save the model file
    save_model(model, save_path)

num_epochs = 5
train_validate(num_epochs)