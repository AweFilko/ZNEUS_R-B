import re
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from numpy.f2py.auxfuncs import throw_error
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from evaluation import *

# ________________________________main material_____________________________________
# Load config file - main material

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
def load_data(cfg):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", str(cfg["setup"]["df_file"]))    )
    if cfg["setup"]["debug"]:
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

def df_sport_adjust(df, cfg):
    sport = cfg["setup"]["sport"]
    df = df.where(df['labels'].isin(sport))
    df = df.dropna()
    if cfg["setup"]["debug"]:
        print("data frame adjustment for :", sport)
    return df

def check_size(df, cfg):
    target_size = cfg["setup"]["img_size"]
    target_size = re.findall(r"\d+", target_size)
    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}")
        if img.size != target_size:
            throw_error(f"{pth} is not a {target_size[0]}x{target_size[1]} image")
    if cfg["setup"]["debug"]:
        print("No size anomalies")

def check_color(df, cfg):
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

def check_duplicates(df, cfg):
    if cfg["setup"]["debug"]:
        print("Duplicate filepaths:", df["filepaths"].duplicated().sum())
        print("Duplicate rows:", df.duplicated().sum())

def check_corrupt_images(df, cfg):
    corrupt = []
    for pth in df["filepaths"]:
        full_path = f"../data/{pth}"
        try:
            img = Image.open(full_path)
            img.verify()
        except (UnidentifiedImageError, OSError):
            corrupt.append(full_path)

    if cfg["setup"]["debug"]:
        print("Corrupt images found:", len(corrupt))
        for c in corrupt:
            print(" -", c)

def check_blank_images(df, cfg ,std_threshold=5):
    blank_images = []

    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}").convert("RGB")
        arr = np.array(img).astype(np.float32)
        if arr.std() < std_threshold:
            blank_images.append(pth)

    if cfg["setup"]["debug"]:
        print("Blank or near-blank images:", len(blank_images))
        for p in blank_images:
            print(" -", p)

def compute_mean_std(df, cfg):
    means = []
    stds = []
    for pth in df["filepaths"]:
        img = Image.open(f"../data/{pth}").convert("RGB")
        img = np.array(img).astype("float32") / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    if cfg["setup"]["debug"]:
        print("Mean and std:", mean, std)
    return mean, std

def build_transform(df, cfg,normalize=True):
    values = cfg['setup']['transform']
    if cfg["setup"]["debug"]:
        print("Building transform")

    transform_list = [
        transforms.RandomHorizontalFlip(float(values['horizontal_flip_prob'])),
        transforms.RandomRotation(int(values['random_rotation_degree'])),
        transforms.ColorJitter(**values['color_jitter']),
        transforms.ToTensor()
    ]

    if normalize:
        mean, std = compute_mean_std(df, cfg)
        transform_list += [transforms.Normalize(mean, std)]

    return transforms.Compose(transform_list)

class SingleImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["filepaths"].values
        self.label_names = df["labels"].values
        self.transform = transform
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


def image_loader(df, cfg):
    batch_size = cfg['model_hyperparams']['batch_size']
    transform = build_transform(df, cfg,normalize=cfg['setup']['transform']['normalization'])

    dataset_test = SingleImageDataset(df[df['data set'] == 'test'], transform)
    dataset_train = SingleImageDataset(df[df['data set'] == 'train'], transform)
    dataset_validate = SingleImageDataset(df[df['data set'] == 'valid'], transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    loader_validate = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True)

    return loader_test, loader_train, loader_validate

# print(loader)

def preprocess_nd_load(cfg):
    df = load_data(cfg=cfg)
    df = df_sport_adjust(df, cfg=cfg)
    dist_plot(df)
    check_size(df, cfg=cfg)
    check_duplicates(df, cfg=cfg)
    check_corrupt_images(df, cfg=cfg)
    check_blank_images(df, cfg=cfg)
    loader = image_loader(df, cfg=cfg)
    return loader


# dist_plot(ld)
# check_size(ld)
# check_color(ld)
# check_duplicates(ld)
# check_corrupt_images(ld)
# check_blank_images(ld)

# print(compute_mean_std(ld))


