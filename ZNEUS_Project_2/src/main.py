import yaml

from preprocess import *
from model import *
from train import *
from evaluation import *


if __name__ == '__main__':

    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    loader = preprocess_nd_load(cfg)

    num_classes = len(cfg['setup']['sport'])
    device = cfg['setup']['device']

    # Train SimpleCNN
    simple = SimpleCNN(num_classes)
    trained_simple, hist_simple = train_model(simple, loader[1], cfg)

    # Train DeepCNN
    deep = DeepCNN(num_classes)
    trained_deep, hist_deep = train_model(deep, loader[1], cfg)
    # Train openCV thingy
    # print("\n=== Train  ===")
    # hereeeee

    # SimpleCNN analysis
    print("\n=== SimpleCNN analysis ===")
    acc_s, prec_s, rec_s, f1_s, y_true_s, y_pred_s = get_metrics(trained_simple, loader[2],device)
    print("\n=== DeepCNN analysis ===")
    plot_training_curves(hist_simple, "SimpleCNN")

    confused_mat(y_true_s, y_pred_s, cfg['setup']['sport'])

    # DeepCNN analysis
    print("\n=== DeepCNN analysis ===")
    acc_d, prec_d, rec_d, f1_d, y_true_d, y_pred_d = get_metrics(trained_deep, loader[2], device)
    plot_training_curves(hist_deep, "DeepCNN")

    confused_mat(y_true_d, y_pred_d, cfg['setup']['sport'])

    # OpenCV analysis
    # print("\n===  analysis ===")

    # Comparison
    compare_models({
        "SimpleCNN": (acc_s, prec_s, rec_s, f1_s),
        "DeepCNN": (acc_d, prec_d, rec_d, f1_d),
    })

    # TODO Validation Metrics: ACC, PRE, REC, F1, Confusion matrix
    # TODO Transformations: RandomHorizontalFlip,  RandomRotation, ColorJitter (done)
    # TODO opencv library, color histogram, entropy statistics, time and acc differences

