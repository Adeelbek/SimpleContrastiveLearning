import os
import urllib.request
from copy import deepcopy
from urllib.error import HTTPError

import matplotlib 
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.types import Device
import torch.utils.data as data 
import torchvision 
from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import STL10
import tqdm
from utils import *
from SimCLR import *
from train import *

from utils import download_pretrained_files

plt.set_cmap("cividis")
# %matplotlib inline
set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.set()

#%load_ext tensorboard

DATASET_PATH = "/home/odilbek/Desktop/SimCLR/data/"

CHECKPOINT_PATH = "/home/odilbek/Desktop/SimCLR/saved_models/ContrastiveLearning/"

NUM_WORKERS = os.cpu_count()

pl.seed_everything(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

def show_samples(demo_samples, unlabeled_data):
    imgs=torch.stack([img for idx in range(demo_samples) for img in unlabeled_data[idx][0]], dim=0) 
    img_grid=torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
    img_grid=img_grid.permute(1,2,0)
    plt.figure(figsize=(10, 5))
    plt.title("Augmented image examples of the STL10 dataset")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()

def plot_results(dataset_sizes, test_score):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(
        dataset_sizes,
        test_scores,
        "--",
        color="#000",
        marker="*",
        markeredgecolor="#000",
        markerfacecolor="y",
        markersize=16,
        )
    plt.xscale("log")
    plt.xticks(dataset_sizes, labels=dataset_sizes)
    plt.title("STL10 classification over dataset size", fontsize=14)
    plt.xlabel("Number of images per class")
    plt.ylabel("Test accuracy")
    plt.minorticks_off()
    plt.show()




if __name__ == "__main__":
    pl.seed_everything(42)
    #download_pretrained_files(saved_dir=CHECKPOINT_PATH)
    train_data, unlabel_data = get_train_unlabeled_data()
    show_samples(demo_samples=6, unlabeled_data=unlabel_data)
    simclr_model = train_simclr(batch_size=256,hiddin_dim=128,lr=5e-4,temperature=0.07,weight_decay=1e-4, max_epochs=500)
    train_img_data, test_img_data = get_train_test_img_data(data_dir=DATASET_PATH)
    train_feats_simclr = prepare_data_features(simclr_model, train_img_data, device=device)
    test_feats_simclr = prepare_data_features(simclr_model, test_img_data, device=device)
    results = {}
    for num_imgs_per_label in [10, 20, 50, 100, 200, 500]:
        sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
        _, small_set_results = train_logreg(
            batch_size=64,
            train_feats_data=sub_train_set,
            test_feats_data=test_feats_simclr,
            model_suffix=num_imgs_per_label,
            feature_dim=train_feats_simclr.tensors[0].shape[1],
            num_classes=10,
            lr=1e-3,
            weight_decay=1e-3,device=device)
        results[num_imgs_per_label] = small_set_results
    dataset_sizes = sorted(k for k in results)
    test_scores = [results[k]["test"] for k in dataset_sizes]
    for k, score in zip(dataset_sizes, test_scores):
        print(f"Test accuracy for {k:3d} images per label: {100*score:4.2f}%")
    plot_results(dataset_sizes=dataset_sizes, test_score=test_scores)
    base_transform = get_baseline_transforms()
    train_img_aug_data = STL10(root=DATASET_PATH, split="train", download=True, transform=base_transform)
    resnet_model, resnet_result = train_resnet(batch_size=64, num_classes=10, lr=1e-3, weight_decay=2e-4, max_epochs=100)
    print(f"Accuracy on training set: {100*resnet_result['train']:4.2f}%")
    print(f"Accuracy on test result: {100*resnet_result['test']:4.2f}%")





