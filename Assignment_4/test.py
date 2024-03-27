from data_process import get_data, get_improved_data
import models

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def show_t_SNE(inputs, model):
    # 获取全连接层的输出
    fc_output = model.get_fc_output(inputs)  # 假设输入数据为`x`
    fc_output = fc_output.cpu().numpy()  # 将张量移动到CPU并转换为NumPy数组

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    fc_output_tsne = tsne.fit_transform(fc_output)

    # 可视化降维结果
    plt.scatter(fc_output_tsne[:, 0], fc_output_tsne[:, 1])
    plt.title("t-SNE Visualization of Fully Connected Layer Output")
    plt.show()

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

def get_model(save_path, model):

    save_file = os.path.join(save_path, 'saved_model.pth')
    model.load_state_dict(torch.load(save_file))
    print(f"Model loaded from {save_file}.")
    return model

def test(model_choice, is_improve):
    save_path = get_dir()
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
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

    # get data
    if is_improve == 0:
        _, _, test_dataset = get_data()
    else:
        _, _, test_dataset = get_improved_data()
    # load the saved model
    model = get_model(save_path, model)
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
            # show_t_SNE(images, model)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

# test code
# test()