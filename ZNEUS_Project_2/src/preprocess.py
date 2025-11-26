import re
import pandas as pd
import os
import yaml
#from PIL import Image, UnidentifiedImageError
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
#from torchvision import transforms
from numpy.f2py.auxfuncs import throw_error
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
#import torch

from model import *
from train import *
from evaluation import *

# ________________________________main material_____________________________________
# Load config file - main material
path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg)

DEBUG = cfg["setup"]["debug"]

# ld = load_data()
# ld = df_sport_adjust(ld)
# dist_plot(ld)
#check_size(ld)
#check_color(ld)
# check_duplicates(ld)
# check_corrupt_images(ld)
# check_blank_images(ld)
# check_aspect_ratio(ld)

#___________________________________________________________________________________

#Load data into data frame
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", str(cfg["setup"]["df_file"]))    )
    if DEBUG:
        print(df.head(10))
        print(df.info())
    return df

def dist_plot(df):
    plt.figure(figsize=(15, 8))
    sns.countplot(data=df, x='labels')
    plt.xticks(rotation=25)
    plt.title("Class distribution")
    plt.show()
    print(df["labels"].value_counts(normalize=True) * 100)

def df_sport_adjust(df):
    sport = cfg["setup"]["sport"]
    df = df.where(df['labels'].isin(sport))
    df = df.dropna()
    if DEBUG:
        print("data frame adjustment for :", sport)
    return df

def check_size(df):
    target_size = cfg["setup"]["img_size"]
    target_size = re.findall(r"\d+", target_size)
    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}")
        if img.size != target_size:
            throw_error(f"{pth} is not a {target_size[0]}x{target_size[1]} image")
    if DEBUG:
        print("No size anomalies")

def check_color(df):
    fig, axes = plt.subplots(2,round(len(df['labels'].unique()) / 2 + 0.1),figsize=(15, 8))
    axes = axes.flatten()
    for i, label in enumerate(cfg["setup"]["sport"]):
        count = 0
        mean_img = np.zeros((224, 224, 3), dtype=np.float32)
        for pth in df[df['labels'] == label]["filepaths"]:
            img = Image.open(f"../data/{pth}").convert("RGB")
            img = np.array(img).astype("float32") / 255
            mean_img += img
            count += 1
        if count > 0:
            mean_img /= count
        ax = axes[i]
        ax.imshow(mean_img)
        ax.set_title(label)
        ax.axis("off")
    for j in range(len(cfg["setup"]["sport"]), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

def check_duplicates(df):
    if DEBUG:
        print("Duplicate filepaths:", df["filepaths"].duplicated().sum())
        print("Duplicate rows:", df.duplicated().sum())

def check_corrupt_images(df):
    corrupt = []
    for pth in df["filepaths"]:
        full_path = f"../data/{pth}"
        try:
            img = Image.open(full_path)
            img.verify()
        except (UnidentifiedImageError, OSError):
            corrupt.append(full_path)

    if DEBUG:
        print("Corrupt images found:", len(corrupt))
        for c in corrupt:
            print(" -", c)

def check_blank_images(df, std_threshold=5):
    blank_images = []

    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}").convert("RGB")
        arr = np.array(img).astype(np.float32)
        if arr.std() < std_threshold:
            blank_images.append(pth)

    if DEBUG:
        print("Blank or near-blank images:", len(blank_images))
        for p in blank_images:
            print(" -", p)

def compute_mean_std(df):
    means = []
    stds = []
    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}").convert("RGB")
        img = np.array(img).astype("float32") / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    if DEBUG:
        print("Mean and std:", mean, std)
    return mean, std

def build_transform(df, normalize=True):
    values = cfg['setup']['transform']
    if DEBUG:
        print("Building transform")

    transform_list = [
        transforms.RandomHorizontalFlip(float(values['horizontal_flip_prob'])),
        transforms.RandomRotation(int(values['random_rotation_degree'])),
        transforms.ColorJitter(**values['color_jitter']),
        transforms.ToTensor()
    ]

    if normalize:
        mean, std = compute_mean_std(df)
        transform_list += [transforms.Normalize(mean, std)]

    return transforms.Compose(transform_list)

# -------------------------------------------------------------------------
# Custom dataset for single image paths
# -------------------------------------------------------------------------
# class SingleImageDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.paths = df["filepaths"].values
#         self.labels = df["labels"].values
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, idx):
#         img_path = f"../data/{self.paths[idx]}"
#         img = Image.open(img_path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         # convert labels to numeric (ImageFolder emulation)
#         return img, self.labels[idx]

class SingleImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["filepaths"].values
        self.label_names = df["labels"].values
        self.transform = transform

        #mapping string to integer
        self.classes = sorted(df["labels"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = f"../data/{self.paths[idx]}"
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_name = self.label_names[idx]
        label = self.class_to_idx[label_name]   # convert to int

        return img, label


# -------------------------------------------------------------------------

def image_loader(df):
    # correct batch size reading
    batch_size = cfg['model_hyperparams']['batch_size']
    transform = build_transform(df, normalize=cfg['setup']['transform']['normalization'])

    dataset = SingleImageDataset(df, transform)
    loader_ = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader_


ld = load_data()
ld = df_sport_adjust(ld)
loader = image_loader(ld)

# print(loader)


# dist_plot(ld)
# check_size(ld)
# check_color(ld)
# check_duplicates(ld)
# check_corrupt_images(ld)
# check_blank_images(ld)

# print(compute_mean_std(ld))


#later for main:!!!
num_classes = 14
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(ld["labels"].unique())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train SimpleCNN
print("\n=== Train SimpleCNN ===")
simple = SimpleCNN(num_classes)
trained_simple, hist_simple = train_model(simple, loader, num_epochs=50, lr=0.001, device=device)

# Train DeepCNN
print("\n=== Train DeepCNN ===")
deep = DeepCNN(num_classes)
trained_deep, hist_deep = train_model(deep, loader, num_epochs=50, lr=0.0005, device=device)

# Train openCV thingy
# print("\n=== Train  ===")
#hereeeee

# SimpleCNN analysis
print("\n=== SimpleCNN analysis ===")
acc_s, prec_s, rec_s, f1_s, y_true_s, y_pred_s = get_metrics(trained_simple, loader)
plot_training_curves(hist_simple, "SimpleCNN")
# confused_mat(y_true_s, y_pred_s, class_names)

# DeepCNN analysis
print("\n=== DeepCNN analysis ===")
acc_d, prec_d, rec_d, f1_d, y_true_d, y_pred_d = get_metrics(trained_deep, loader)
plot_training_curves(hist_deep, "DeepCNN")
# confused_mat(y_true_d, y_pred_d, class_names)

# OpenCV analysis
# print("\n===  analysis ===")

# Comparison
compare_models({
    "SimpleCNN": (acc_s, prec_s, rec_s, f1_s),
    "DeepCNN":   (acc_d, prec_d, rec_d, f1_d),
})




# TODO Validation Metrics: ACC, PRE, REC, F1, Confusion matrix
# TODO Transformations: RandomHorizontalFlip,  RandomRotation, ColorJitter (done)
# TODO opencv library, color histogram, entropy statistics, time and acc differences
