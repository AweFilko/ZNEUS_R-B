import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

import joblib


def run_feature_selection(X, y, feature_names=None, outdir="./fs_outputs", select_k=100):
    import os
    os.makedirs(outdir, exist_ok=True)

    print("Input shape:", X.shape)

    # -----------------------------
    # 1) Standardization
    # -----------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, f"{outdir}/scaler.joblib")

    # -----------------------------
    # 2) MUTUAL INFORMATION
    # -----------------------------
    vt = VarianceThreshold(threshold=1e-5)
    Xs = vt.fit_transform(Xs)
    joblib.dump(vt, f"{outdir}/variance_filter.joblib")

    if feature_names is not None:
        feature_names = np.array(feature_names)[vt.get_support()]

    joblib.dump(vt, f"{outdir}/variance_threshold.joblib")

    mi = mutual_info_classif(Xs, y, random_state=0)

    print("Xs shape:", Xs.shape)
    print("Feature names length:", len(feature_names))

    mi_df = pd.DataFrame({
        "feature": feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])],
        "mi": mi
    }).sort_values("mi", ascending=False)

    mi_df.to_csv(f"{outdir}/mutual_info_scores.csv", index=False)

    # Plot top MI features
    top = min(30, len(mi))
    plt.figure(figsize=(6, 6))
    plt.barh(mi_df["feature"][:top][::-1], mi_df["mi"][:top][::-1])
    plt.xlabel("Mutual Information")
    plt.title("Top Mutual Information Features")
    plt.tight_layout()
    plt.savefig(f"{outdir}/top_mutual_info.png")
    plt.close()

    # SelectKBest
    selector = SelectKBest(mutual_info_classif, k=min(select_k, X.shape[1]))
    X_mi = selector.fit_transform(Xs, y)
    selected_mask = selector.get_support()
    selected_features = np.array(
        feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    )[selected_mask]

    pd.DataFrame({"selected_features": selected_features}) \
        .to_csv(f"{outdir}/selected_mi_features.csv", index=False)

    joblib.dump(selector, f"{outdir}/mi_selector.joblib")

    print("MI reduced shape:", X_mi.shape)

    # -----------------------------
    # 3) PCA
    # -----------------------------
    pca = PCA().fit(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    plt.figure()
    plt.plot(cum_var)
    plt.axhline(0.90)
    plt.axhline(0.95)
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.savefig(f"{outdir}/pca_scree.png")
    plt.close()

    n90 = np.searchsorted(cum_var, 0.90) + 1
    n95 = np.searchsorted(cum_var, 0.95) + 1

    print("PCA 90% components:", n90)
    print("PCA 95% components:", n95)

    pca_red = PCA(n_components=n95)
    X_pca = pca_red.fit_transform(Xs)

    joblib.dump(pca_red, f"{outdir}/pca_n95.joblib")

    pd.DataFrame({
        "pc": np.arange(1, len(cum_var)+1),
        "explained": pca.explained_variance_ratio_,
        "cumulative": cum_var
    }).to_csv(f"{outdir}/pca_explained_variance.csv", index=False)

    print("PCA reduced shape:", X_pca.shape)



    # -----------------------------
    # 4) QUICK MODEL SANITY CHECK (FENN-LIKE)
    # -----------------------------
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    results = {}

    def cv_score(Z):
        acc = []
        for tr, te in cv.split(Z, y):
            scaler = StandardScaler()
            Ztr = scaler.fit_transform(Z[tr])
            Zte = scaler.transform(Z[te])

            clf = LogisticRegression(max_iter=2000)
            clf.fit(Ztr, y[tr])
            acc.append(clf.score(Zte, y[te]))
        return np.mean(acc)

    results["raw"] = cv_score(Xs)
    results["mutual_info"] = cv_score(X_mi)
    results["pca_95"] = cv_score(X_pca)

    pd.DataFrame([results]).to_csv(f"{outdir}/quick_cv_results.csv", index=False)
    print("CV results:", results)

    # -----------------------------
    # 5) PCA VISUALIZATION
    # -----------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X2 = PCA(n_components=2).fit_transform(Xs)

    plt.figure(figsize=(6, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=y_enc, cmap="tab10", s=12)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection (Colored by Class)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/pca_projection.png")
    plt.close()

    print("Feature selection completed.\n")


