from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset


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

# Choice part:
def get_improved_data():
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotate randomly up to 10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # Adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',  # save path
        train=True,  # load dataset
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


def get_K_data():
    # prepare data for k-fold validation:
    # get data from internet:
    transform = transforms.Compose([
        transforms.ToTensor(),  # transform the img
        transforms.Normalize((0.5,), (0.5,))  # normalize the img
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',  # save path
        train=True,  # load dataset
        download=True,  # auto download
        transform=transform  # transform data
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # merge 2 datasets into K dataset
    K_dataset = ConcatDataset([train_dataset, test_dataset])
    return K_dataset



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