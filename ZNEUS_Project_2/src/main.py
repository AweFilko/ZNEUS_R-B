import wandb
import yaml

from preprocess import *
from train import *
from evaluation import *

from feature_selection import run_feature_selection
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder


def choose_model(config, num_c):
    mode = int(config["model"])
    if mode == 0:
        return SCNN(cfg=config, num_classes=num_c)
    elif mode == 1:
        return DCNN(cfg=config, num_classes=num_c)
    elif mode == 2:
        return ResNet(cfg=config, num_classes=num_c)
    elif mode == 3:
        pca = joblib.load("./fs_outputs/pca_n95.joblib")
        return FENN(input_dim=pca.n_components_, cfg=config, num_classes=num_c)

    else:
        raise Exception("Unknown mode")



if __name__ == '__main__':

    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    mode = int(cfg["model"])

    # =============================
    # CNN MODELS → IMAGE PIPELINE
    # =============================
    if mode in [0, 1, 2]:  # SCNN, DCNN, ResNet
        loader = preprocess_nd_load(cfg)

    # =============================
    # FENN MODEL → FEATURE PIPELINE
    # =============================
    elif mode == 3:
        print("Using FEATURE pipeline for FENN")

        # 1) Load raw dataframe
        df = load_data(cfg)
        df = df_sport_adjust(df, cfg=cfg)

        # 2) Build features (your existing function)
        df = fe_build_df(df)

        # 3) Separate X, y
        X = df.drop(columns=["label"]).values
        y = df["label"].values
        le = LabelEncoder()
        y = le.fit_transform(y)

        joblib.dump(le, "./fs_outputs/label_encoder.joblib")

        # ===============================
        # CLEAN SPLIT BEFORE PCA
        # ===============================
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 4) RUN FEATURE SELECTION (TRAIN ONLY)
        run_feature_selection(
            X_train,
            y_train,
            feature_names=df.drop(columns=["label"]).columns.tolist(),
            outdir="./fs_outputs",
            select_k=100
        )

        # 5) LOAD PCA MODEL (FIT ON TRAIN ONLY)
        scaler = joblib.load("./fs_outputs/scaler.joblib")
        vt = joblib.load("./fs_outputs/variance_filter.joblib")
        pca = joblib.load("./fs_outputs/pca_n95.joblib")

        X_train_scaled = scaler.transform(X_train)
        X_train_vt = vt.transform(X_train_scaled)
        X_train_final = pca.transform(X_train_vt)

        X_val_scaled = scaler.transform(X_val)
        X_val_vt = vt.transform(X_val_scaled)
        X_val_final = pca.transform(X_val_vt)

        # 6) Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        loader = [
            None,
            DataLoader(train_dataset, batch_size=cfg["model_hyperparams"]["batch_size"], shuffle=True),
            DataLoader(val_dataset, batch_size=cfg["model_hyperparams"]["batch_size"], shuffle=False),
        ]


    num_classes = len(cfg['setup']['sport'])
    device = cfg['setup']['device']

    # wandb.login()
    #
    model = choose_model(cfg, num_classes)
    #
    or_name = get_original_name(model)

    # run = wandb.init(project="ZNEUS_R&B", config=cfg, tags=[or_name])
    # wandb.run.name = f"{run.name}_{or_name}"
    # print(f"\n==={run.name} Train===")

    trained_model, hist_model = simple_train(model, loader[1], cfg)\
        if int(cfg["train"])\
        else complex_train(model, loader[1],loader[2], cfg)

    # print(f"\n==={run.name} Analysis===")
    evaluate_model(trained_model, loader[2], device, cfg)
    # wandb.finish()

    # TODO opencv library, color histogram, entropy statistics, time and acc differences

# if __name__ == '__main__':
#     path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
#     with open(path, "r") as f:
#         cfg = yaml.safe_load(f)
#     print(cfg)
#
#     ld = load_data(cfg)
#     ld = df_sport_adjust(ld, cfg=cfg)
#
#     ld = fe_build_df(ld)
#     print(ld.info())
#
#     corr = ld.drop(columns=["label"]).corr().abs()
#     mean_corr = corr.mean()
#     print(mean_corr.sort_values(ascending=False).head(20))