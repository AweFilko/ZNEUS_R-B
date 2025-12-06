import yaml

from preprocess import *
from train import *
from evaluation import *
import joblib


def choose_model(config, num_c):
    mode = int(config["model"])
    if mode == 0:
        return SCNN(cfg=config, num_classes=num_c)
    elif mode == 1:
        return DCNN(cfg=config, num_classes=num_c)
    elif mode == 2:
        return ResNet(cfg=config, num_classes=num_c)
    elif mode == 3:
        pca = joblib.load("../fs_outputs/pca_n95.joblib")
        return FENN(input_dim=pca.n_components_, cfg=config, num_classes=num_c)
    else:
        raise Exception("Unknown mode")



if __name__ == '__main__':

    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    mode = int(cfg["model"])
    loader = None

    if mode in [0, 1, 2]:  # SCNN, DCNN, ResNet
        loader = preprocess_nd_load(cfg)

    elif mode == 3:
        loader = preprocess_nd_load_fe(cfg)


    num_classes = len(cfg['setup']['sport'])
    device = cfg['setup']['device']

    wandb.login()

    model = choose_model(cfg, num_classes)
    or_name = get_original_name(model)

    run = wandb.init(project="ZNEUS_R&B", config=cfg, tags=[or_name])
    wandb.run.name = f"{run.name}_{or_name}"
    print(f"\n==={run.name} Train===")

    trained_model, hist_model = simple_train(model, loader[0], cfg)\
        if int(cfg["train"])\
        else complex_train(model, loader[0],loader[1], cfg)

    print(f"\n==={run.name} Analysis===")
    evaluate_model(trained_model, (loader[1],loader[2]), device, cfg)
    wandb.finish()