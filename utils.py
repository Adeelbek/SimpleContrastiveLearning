import os 
import urllib.request 
from copy import deepcopy
from urllib.error import HTTPError
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm

NUM_WORKERS = os.cpu_count()


def download_pretrained_files(saved_dir = "/home/odilbek/Desktop/SimCLR/saved_models/ContrastiveLearning/", 
    base_url="https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial1"):
    pretrained_files = [
    "SimCLR.ckpt",
    "ResNet.ckpt",
    "tensorboards/SimCLR/events.out.tfevents.SimCLR",
    "tensorboards/classification/ResNet/events.out.tfevents.ResNet",
    ]
    pretrained_files += [f"LogisticRegression_{size}.ckpt" for size in [10, 20, 50, 100, 200, 500]]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(saved_dir, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(saved_dir, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",e)
    #return pretrained_files

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def get_contrastive_transformation():
    contrast_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], 
                               p=0.8), 
        transforms.RandomGrayscale(p=0.2), 
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5)),
    ]
    )
    return contrast_transform

def get_train_unlabeled_data(data_dir="data/", n_views=2):
    con_trans = get_contrastive_transformation()
    train_data = STL10(root=data_dir, split="train", download=True, transform=ContrastiveTransformations(con_trans, n_views=n_views))
    unlabeled_data = STL10(root=data_dir, split="unlabeled", download=True, transform=ContrastiveTransformations(con_trans, n_views=n_views))
    return train_data,unlabeled_data

def get_train_test_img_data(data_dir=""):
    img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
    train_img_data = STL10(root=data_dir, split="train", download=True, transform=img_transforms)
    test_img_data = STL10(root=data_dir, split="test", download=True, transform=img_transforms)
    print("Number of training examples:", len(train_img_data))
    print("Number of test examples:", len(test_img_data))
    return train_img_data, test_img_data


@torch.no_grad()
def prepare_data_features(model, dataset, device):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm.tqdm_notebook(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]
    return data.TensorDataset(feats, labels)

def get_baseline_transforms():
    return transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop(size=96, scale=(0.8,1.0)),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)),])
    
def get_smaller_dataset(original_dataset,num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *(t.unflatten(0, (10, 500))[:, :num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors)
    )
    return new_dataset




