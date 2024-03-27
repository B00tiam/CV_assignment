from data_process import get_data, get_improved_data
import models

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

def paint(lr_his, train_loss_list, train_acc_list, val_loss_list, val_acc_list, save_path):
    # draw 2 graphs
    plt.subplot(121)
    plt.plot(train_acc_list[:], '-o', label="train_acc")
    plt.plot(val_acc_list[:], '-o', label="val_acc")
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(train_loss_list[:], '-o', label="train_loss")
    plt.plot(val_loss_list[:], '-o', label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(save_path + '/acc_loss.png')
    plt.show()

    # draw lr graph
    plt.figure(figsize=(10, 6))
    plt.plot(lr_his, marker='o', linestyle='-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.grid(True)
    plt.savefig(save_path + '/lr.png')
    plt.show()

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

def save_log(epoch, train_loss, train_acc, val_loss, val_acc, save_path, model_name, best_epoch, best_acc, cur_lr):
    # logging
    log_path = os.path.join(save_path, 'training.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Training model: {model_name}')
    logging.info(f'Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Current lr: {cur_lr}')
    logging.info(f'The best Accuracy: {best_acc}, which belongs to Epoch [{best_epoch + 1}]')

def train_validate(num_epochs, model_choice, is_lr, is_improve):
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # get data
    if is_improve == 0:
        train_dataset, val_dataset, _ = get_data()
    else:
        train_dataset, val_dataset, _ = get_improved_data()
    # get loader
    # train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # validate loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # initialize model
    if model_choice == 0:
        # baseline
        model = models.LeNet5_baseline().to(device)
    elif model_choice == 1:
        # var1
        model = models.LeNet5_var1().to(device)
    elif model_choice == 2:
        # var2
        model = models.LeNet5_var2().to(device)
    elif model_choice == 3:
        # var3
        model = models.LeNet5_var3().to(device)
    else:
        # var4
        model = models.LeNet5_var4().to(device)
    model_name = model.__class__.__name__
    # Summary the model
    summary(model, (1, 28, 28))

    # Model save path
    save_path = get_dir()

    # loss func and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    lr_his = []

    # Best model according to the best accuracy on validation dataset
    best_acc = 0.0
    best_model = None
    best_epoch = 0

    # train & validate
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):

        model, train_loss, train_acc = train(model, train_loader, device, optimizer, loss_func)
        model, val_loss, val_acc = validate(model, val_loader, device, loss_func)
        # learning rate
        cur_lr = optimizer.param_groups[0]['lr']
        lr_his.append(cur_lr)
        if is_lr == 1:
            scheduler.step()
        # compare the accuracy on validation:
        if val_acc > best_acc:
            best_model = model
            best_acc = val_acc
            best_epoch = epoch

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Current lr: {cur_lr}")
        save_log(epoch, train_loss, train_acc, val_loss, val_acc, save_path, model_name, best_epoch, best_acc, cur_lr)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

    # Save the best model file
    save_model(best_model, save_path)
    # Paint the chart
    paint(lr_his, train_loss_list, train_acc_list, val_loss_list, val_acc_list, save_path)

# test code
# num_epochs = 15
# train_validate(num_epochs)