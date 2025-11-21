import pandas as pd
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
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

#___________________________________________________________________________________

#Load data into data frame
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", str(cfg["setup"]["df_file"])))
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

def df_sport_adjust(df):
    sport = cfg["setup"]["sport"]
    df = df.where(df['labels'].isin(sport))
    df = df.dropna()
    return df

def check_size(df):
    for path in df["filepaths"]:
        img = Image.open(f"../data/{path}")
        if img.size != (224, 224):
            throw_error(f"{path} is not a 224x224 image")

def check_color(df):
    for label in cfg["setup"]["sport"]:
        count = 0
        mean_img = np.zeros((224, 224, 3))
        for path in df[df['labels'] == label]["filepaths"]:
            count += 1
            img = Image.open(f"../data/{path}").convert("RGB")
            img = np.array(img).astype("float32") / 255
            mean_img += img

        mean_img /= count
        plt.figure(figsize=(7, 2))
        plt.imshow(mean_img)
        plt.title(f"Mean Image: {label}")
        plt.axis("off")
        plt.show()



