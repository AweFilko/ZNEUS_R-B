import pandas as pd
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from numpy.f2py.auxfuncs import throw_error
import numpy as np

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
    return df

def check_size(df):
    target_size = cfg["setup"]["img_size"]
    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}")
        if img.size != target_size:
            throw_error(f"{pth} is not a {target_size[0]}x{target_size[1]} image")
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
    return np.mean(means, axis=0), np.mean(stds, axis=0)

def build_transform(df, normalize=True):
    transform_list = [transforms.ToTensor()]

    if normalize:
        mean, std = compute_mean_std(df)
        transform_list += transforms.Normalize(mean, std)

    return transforms.Compose(transform_list)


ld = load_data()
ld = df_sport_adjust(ld)
# dist_plot(ld)
# check_size(ld)
# check_color(ld)
# check_duplicates(ld)
# check_corrupt_images(ld)
# check_blank_images(ld)
# check_aspect_ratio(ld)

#print(compute_mean_std(ld))

# TODO Validation Metrics: ACC, PRE, REC, F1, Confusion matrix

# TODO Transformations: RandomHorizontalFlip,  RandomRotation, ColorJitter

# TODO opencv library, color histogram, entropy statistics, time and acc differences
