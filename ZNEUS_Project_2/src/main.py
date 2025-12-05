import yaml

from preprocess import *
from train import *
from evaluation import *

def choose_model(config, num_c):
    mode = int(config["model"])
    if mode == 0:
        return SCNN(cfg=config, num_classes=num_c)
    elif mode == 1:
        return DCNN(cfg=config, num_classes=num_c)
    elif mode == 2:
        return ResNet(cfg=config, num_classes=num_c)
    elif mode == 3:
        return FENN(cfg=config, num_classes=num_c)
    else:
        raise Exception("Unknown mode")



if __name__ == '__main__':

    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    loader = preprocess_nd_load(cfg)

    num_classes = len(cfg['setup']['sport'])
    device = cfg['setup']['device']

    wandb.login()

    model = choose_model(cfg, num_classes)

    or_name = get_original_name(model)

    run = wandb.init(project="ZNEUS_R&B", config=cfg, tags=[or_name])
    wandb.run.name = f"{run.name}_{or_name}"
    print(f"\n==={run.name} Train===")

    trained_model, hist_model = simple_train(model, loader[1], cfg)\
        if int(cfg["train"])\
        else complex_train(model, loader[1],loader[2], cfg)

    print(f"\n==={run.name} Analysis===")
    evaluate_model(trained_model, loader[2], device, cfg)
    wandb.finish()

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