import pandas as pd
import os
import yaml


# ________________________________main material_____________________________________
# Load config file - main material
path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg)

DEBUG = cfg["setup"]["debug"]

#ld = load_data()

#___________________________________________________________________________________

#Load data into data frame
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", str(cfg["setup"]["df_file"])))
    if DEBUG:
        print(df.head(10))
    return df







