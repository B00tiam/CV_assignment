from data_process import get_data
import models

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim


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

    new_dir_num = new_dir_num - 1
    # create the subdirectory for saving the trained model
    sub_dir = f'{new_dir_num:02}'
    save_path = os.path.join(save_dir, sub_dir)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def get_model(save_path):
    model = models.LeNet5_baseline().to(device)
    save_file = os.path.join(save_path, 'saved_model.pth')
    model.load_state_dict(torch.load(save_file))
    print(f"Model loaded from {save_file}.")
    return model

def test():
    save_path = get_dir()
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # get data
    _, _, test_dataset = get_data()
    # load the saved model
    model = get_model(save_path)
    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # test the model on test dataset:
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

# test code
test()