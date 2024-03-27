from data_process import get_K_data
from train_validate import train, validate
import models

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

def cross_validation(K, num_epochs, model_choice):
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # get data
    K_dataset = get_K_data()

    # Best model list
    best_acc_list = []
    best_epoch_list = []

    # k-fold cross-validation
    kf = KFold(n_splits=K, shuffle=True)
    fold = 0
    for train_index, val_index in kf.split(K_dataset):
        fold = fold + 1
        print(f"Fold: {fold}")
        train_subset = torch.utils.data.Subset(K_dataset, train_index)
        val_subset = torch.utils.data.Subset(K_dataset, val_index)

        # get loader
        # train loader
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
        # validate loader
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

        # Best model according to the best accuracy on validation dataset
        best_acc = 0.0
        best_epoch = 0

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

        # loss func and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # train & validate
        for epoch in range(num_epochs):
            model, train_loss, train_acc = train(model, train_loader, device, optimizer, loss_func)
            model, val_loss, val_acc = validate(model, val_loader, device, loss_func)
            # compare the accuracy on validation:
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
            print(f"Fold: {fold}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # save the best performance of each fold:
        print(f"The best epoch is: Epoch [{best_epoch + 1}], where the best accuracy is: [{best_acc}]")
        best_acc_list.append(best_acc)
        best_epoch_list.append(best_epoch)

    print(best_epoch_list)
    print(best_acc_list)