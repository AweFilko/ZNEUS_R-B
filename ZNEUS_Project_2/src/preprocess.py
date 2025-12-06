import re
import os

import joblib
import pandas as pd
from PIL import Image, UnidentifiedImageError

from numpy.f2py.auxfuncs import throw_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from evaluation import *
from extraction import *

def load_data(cfg):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", str(cfg["setup"]["df_file"]))    )
    if cfg["setup"]["debug"]:
        print(df.head(10))
        print(df.info())
    return df

def dist_plot(df, cfg):
    if cfg["setup"]["debug"]:
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
    if cfg["setup"]["debug"]:
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
    transform_eval = [transforms.ToTensor()]

    if normalize:
        mean, std = compute_mean_std(df, cfg)
        transform_list += [transforms.Normalize(mean, std)]
        transform_eval += [transforms.Normalize(mean, std)]

    return transforms.Compose(transform_list), transforms.Compose(transform_eval)

def run_feature_selection(x, y, feature_names=None, out_directory="../fs_outputs", select_k=100, cfg=None):
    import os
    os.makedirs(out_directory, exist_ok=True)

    print("Input shape:", x.shape)

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    joblib.dump(scaler, f"{out_directory}/scaler.joblib")

    vt = VarianceThreshold(threshold=1e-5)
    xs = vt.fit_transform(xs)
    joblib.dump(vt, f"{out_directory}/variance_filter.joblib")

    if feature_names is not None:
        feature_names = np.array(feature_names)[vt.get_support()]

    joblib.dump(vt, f"{out_directory}/variance_threshold.joblib")

    mi = mutual_info_classif(xs, y, random_state=int(cfg['setup']['random_state']))

    print("xs shape:", xs.shape)
    print("Feature names length:", len(feature_names))

    mi_df = pd.DataFrame({
        "feature": feature_names if feature_names is not None else [f"f{i}" for i in range(x.shape[1])],
        "mi": mi
    }).sort_values("mi", ascending=False)

    mi_df.to_csv(f"{out_directory}/mutual_info_scores.csv", index=False)

    # Plot top MI features
    # top = min(30, len(mi))
    # plt.figure(figsize=(6, 6))
    # plt.barh(mi_df["feature"][:top][::-1], mi_df["mi"][:top][::-1])
    # plt.xlabel("Mutual Information")
    # plt.title("Top Mutual Information Features")
    # plt.tight_layout()
    # plt.savefig(f"{out_directory}/top_mutual_info.png")
    # plt.close()

    # SelectKBest
    selector = SelectKBest(mutual_info_classif, k=min(select_k, x.shape[1]))
    x_mi = selector.fit_transform(xs, y)
    selected_mask = selector.get_support()
    selected_features = np.array(
        feature_names if feature_names is not None else [f"f{i}" for i in range(x.shape[1])]
    )[selected_mask]

    pd.DataFrame({"selected_features": selected_features}) \
        .to_csv(f"{out_directory}/selected_mi_features.csv", index=False)

    joblib.dump(selector, f"{out_directory}/mi_selector.joblib")

    print("MI reduced shape:", x_mi.shape)

    pca = PCA().fit(xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # plt.figure()
    # plt.plot(cum_var)
    # plt.axhline(0.90)
    # plt.axhline(0.95)
    # plt.xlabel("Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title("PCA Scree Plot")
    # plt.grid(True)
    # plt.savefig(f"{out_directory}/pca_scree.png")
    # plt.close()

    n90 = np.searchsorted(cum_var, 0.90) + 1
    n95 = np.searchsorted(cum_var, 0.95) + 1

    print("PCA 90% components:", n90)
    print("PCA 95% components:", n95)

    pca_red = PCA(n_components=n95)
    x_pca = pca_red.fit_transform(xs)

    joblib.dump(pca_red, f"{out_directory}/pca_n95.joblib")

    pd.DataFrame({
        "pc": np.arange(1, len(cum_var)+1),
        "explained": pca.explained_variance_ratio_,
        "cumulative": cum_var
    }).to_csv(f"{out_directory}/pca_explained_variance.csv", index=False)

    print("PCA reduced shape:", x_pca.shape)

    cv = StratifiedKFold(5, shuffle=True, random_state=int(cfg['setup']['random_state']))
    results = {}

    def cv_score(z):
        acc = []
        for tr, te in cv.split(z, y):
            scaler = StandardScaler()
            ztr = scaler.fit_transform(z[tr])
            zte = scaler.transform(z[te])

            clf = LogisticRegression(max_iter=2000)
            clf.fit(ztr, y[tr])
            acc.append(clf.score(zte, y[te]))
        return np.mean(acc)

    results["raw"] = cv_score(xs)
    results["mutual_info"] = cv_score(x_mi)
    results["pca_95"] = cv_score(x_pca)

    pd.DataFrame([results]).to_csv(f"{out_directory}/quick_cv_results.csv", index=False)
    print("CV results:", results)

    # le = LabelEncoder()
    # y_enc = le.fit_transform(y)

    # x2 = PCA(n_components=2).fit_transform(xs)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(x2[:, 0], x2[:, 1], c=y_enc, cmap="tab10", s=12)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("PCA 2D Projection (Colored by Class)")
    # plt.tight_layout()
    # plt.savefig(f"{out_directory}/pca_projection.png")
    # plt.close()

    print("Feature selection completed.\n")

class SingleImageDataset(Dataset):
    def __init__(self, df, transform=None, extractor=None, class_to_idx=None):
        self.paths = df["filepaths"].values
        self.label_names = df["labels"].values
        self.transform = transform
        self.extractor = extractor

        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = f"../data/{self.paths[idx]}"
        img = Image.open(img_path).convert("RGB")
        if self.extractor:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.transform:
            img = self.transform(img)
        elif self.extractor:
            img = self.extractor.extract(img)
            img = torch.tensor(img, dtype=torch.float32)

        label_name = self.label_names[idx]
        label = self.class_to_idx[label_name]   # convert to int

        return img, label


def image_loader(df, cfg):
    batch_size = cfg['model_hyperparams']['batch_size']

    extract = None
    transform = None
    transform_tv = None

    if int(cfg["model"]) == 3:
        extract =  FeatureExtractor()
    else:
        transform,transform_tv = build_transform(df, cfg,
                                                 normalize=cfg['setup']['transform']['normalization'])

    all_classes = sorted(df["labels"].unique())
    global_class_to_idx = {c: i for i, c in enumerate(all_classes)}

    dataset_test = SingleImageDataset(df[df['data set'] == 'test'],
                                      transform=transform_tv,
                                      extractor= extract,
                                      class_to_idx=global_class_to_idx)
    dataset_train = SingleImageDataset(df[df['data set'] == 'train'],
                                       transform=transform,
                                       extractor= extract,
                                       class_to_idx=global_class_to_idx)
    dataset_validate = SingleImageDataset(df[df['data set'] == 'valid'],
                                          transform=transform_tv,
                                          extractor= extract,
                                          class_to_idx=global_class_to_idx)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    loader_validate = DataLoader(dataset_validate, batch_size=batch_size, shuffle=False, drop_last=True)

    return loader_train, loader_validate, loader_test

def fe_build_df(df, cfg = None):
    if not cfg['setup']['build_fe']:
        return pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        "..", "data","feature_extracted.csv"))
    extractor = FeatureExtractor()

    feature_vectors = []
    label_list = []
    dataset_type = []

    for idx, row in df.iterrows():
        pth = row["filepaths"]

        img = Image.open(f"../data/{pth}").convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        feats = extractor.extract(img)
        feature_vectors.append(feats)

        label_list.append(row["labels"])
        dataset_type.append(row["data set"])

    x = pd.DataFrame(feature_vectors)
    x.columns = [f"f{i}" for i in range(x.shape[1])]

    x["label"] = label_list
    x["set"] = dataset_type
    x.to_csv("../data/feature_extracted.csv", index=False)

    return x

def preprocess_nd_load(cfg):
    df = load_data(cfg=cfg)
    df = df_sport_adjust(df, cfg=cfg)
    dist_plot(df, cfg=cfg)
    check_size(df, cfg=cfg)
    check_duplicates(df, cfg=cfg)
    check_color(df, cfg=cfg)
    check_corrupt_images(df, cfg=cfg)
    check_blank_images(df, cfg=cfg)
    loader = image_loader(df, cfg=cfg)
    print("Preprocessing done")
    return loader

def preprocess_nd_load_fe(cfg):
    df = load_data(cfg=cfg)
    df = df_sport_adjust(df, cfg=cfg)
    df = fe_build_df(df, cfg=cfg)
    le = LabelEncoder()

    if cfg['setup']['build_fe']:
        x = df.drop(columns=["label", "set"]).values
        y = df["label"].values
        y = le.fit_transform(y)

        joblib.dump(le, "../fs_outputs/label_encoder.joblib")

        run_feature_selection(
            x,
            y,
            feature_names=df.drop(columns=["label", "set"]).columns.tolist(),
            out_directory="../fs_outputs",
            select_k=100,
            cfg=cfg
        )

    x = df[df["set"] == "train"].drop(columns=["label", "set"]).values
    y = df[df["set"] == "train"]["label"].values
    y = le.fit_transform(y)

    x_val = df[df["set"] == "valid"].drop(columns=["label", "set"]).values
    y_val = df[df["set"] == "valid"]["label"].values
    y_val = le.fit_transform(y_val)

    x_test = df[df["set"] == "valid"].drop(columns=["label", "set"]).values
    y_test = df[df["set"] == "valid"]["label"].values
    y_test = le.fit_transform(y_test)

    scaler = joblib.load("../fs_outputs/scaler.joblib")
    vt = joblib.load("../fs_outputs/variance_filter.joblib")
    pca = joblib.load("../fs_outputs/pca_n95.joblib")

    x_trained_scaled = scaler.transform(x)
    x_trained_vt = vt.transform(x_trained_scaled)
    x_train_final = pca.transform(x_trained_vt)

    x_val_scaled = scaler.transform(x_val)
    x_val_vt = vt.transform(x_val_scaled)
    x_val_final = pca.transform(x_val_vt)

    x_test_scaled = scaler.transform(x_test)
    x_test_vt = vt.transform(x_test_scaled)
    x_test_final = pca.transform(x_test_vt)

    x_train_tensor = torch.tensor(x_train_final, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    x_val_tensor = torch.tensor(x_val_final, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    x_test_tensor = torch.tensor(x_test_final, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    loader = [
        DataLoader(train_dataset, batch_size=cfg["model_hyperparams"]["batch_size"], shuffle=True),
        DataLoader(val_dataset, batch_size=cfg["model_hyperparams"]["batch_size"], shuffle=False),
        DataLoader(test_dataset, batch_size=cfg["model_hyperparams"]["batch_size"], shuffle=False),
    ]
    return loader




