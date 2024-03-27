from data_process import get_data, get_improved_data
from test import get_dir, get_model
import models

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def paint_tsne(tsne_fc, label_np, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_fc[:, 0], tsne_fc[:, 1], c=label_np)
    plt.title("t-SNE Visualization of Fully Connected Layer Output")
    # save
    plt.savefig(save_path + '/tsne_fc.png')
    plt.show()

def calculate_tSNE(model_choice, is_improve):
    save_path = get_dir()
    # check if the device use GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # get data
    if is_improve == 0:
        _, _, test_dataset = get_data()
    else:
        _, _, test_dataset = get_improved_data()
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
    # load the saved model
    model = get_model(save_path, model)
    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    feature_list = []
    label_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            fc_outputs = model.get_fc_output(images)
            feature_list.append(fc_outputs)
            label_list.append(labels)

    feature_list = torch.cat(feature_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    feature_np = feature_list.view(feature_list.size(0), -1).cpu().numpy()
    label_np = label_list.view(label_list.size(0), -1).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_fc = tsne.fit_transform(feature_np)

    # draw t-SNE graph
    paint_tsne(tsne_fc, label_np, save_path)