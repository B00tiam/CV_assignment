import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def divide_data(train_dataset):
    # divide the train dataset:
    train_size = int(0.8 * len(train_dataset))  # preserve 80% of train data
    val_size = len(train_dataset) - train_size  # remaining as validate data
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset

def get_data():
    # get data from internet:
    transform = transforms.Compose([
        transforms.ToTensor(),  # transform the img
        transforms.Normalize((0.5,), (0.5,))  # normalize the img
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',  # save path
        train=True,     # load dataset
        download=True,  # auto download
        transform=transform  # transform data
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # divide the data
    train_dataset, val_dataset = divide_data(train_dataset)

    return train_dataset, val_dataset, test_dataset


# train loader
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test loader
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# get images & labels
'''
train_images = train_dataset.data
train_labels = train_dataset.targets

test_images = test_dataset.data
test_labels = test_dataset.targets
'''

# test code
'''
train_dataset, val_dataset, test_dataset = get_data()
# train loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
print(len(train_dataset))
print(len(train_loader))
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    
'''