import os, ast, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tslearn.metrics as ts_metrics

# --- 1. Configuration ---
CFG = {
    "data_dir": "RESULT/JLJG/qijiang/",
    "save_dir": "RESULT/JLJG/qijiang/",
    "n_clusters": 3,
    "weight_rain": 0.6,
    "seed": 42
}
os.makedirs(CFG["save_dir"], exist_ok=True)


# --- 2. Core Logic ---
def run_pipeline():
    # Load and Parse Data
    def load_seq(path):
        df = pd.read_csv(path)
        return df, np.array(df.iloc[:, 2].apply(ast.literal_eval).tolist())

    flow_df, flow_seq = load_seq(f"{CFG['data_dir']}qijiang flow.csv")
    rain_df, rain_seq = load_seq(f"{CFG['data_dir']}qijiang rainfall.csv")

    # Step A: K-Medoids Clustering with DTW
    print("Clustering via DTW...")
    weighted_data = flow_seq + (rain_seq * CFG["weight_rain"])
    dist_matrix = ts_metrics.cdist_dtw(weighted_data)

    km = KMedoids(n_clusters=CFG["n_clusters"], random_state=CFG["seed"], metric="precomputed")
    labels = km.fit_predict(dist_matrix)

    # Step B: SVC Classification
    print("Training SVC...")
    X = np.hstack([flow_seq, rain_seq])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, stratify=labels)

    grid = GridSearchCV(SVC(probability=True), {'C': [1, 10], 'kernel': ['rbf', 'linear']}, cv=3)
    grid.fit(x_train, y_train)
    print(f"Best SVC Params: {grid.best_params_}\n", classification_report(y_test, grid.predict(x_test)))

    # Step C: Export Results
    joblib.dump(grid.best_estimator_, f"{CFG['save_dir']}svc_model.pkl")
    joblib.dump(scaler, f"{CFG['save_dir']}scaler.pkl")

    with pd.ExcelWriter(f"{CFG['save_dir']}flood_clusters.xlsx") as writer:
        for i in range(CFG["n_clusters"]):
            mask = (labels == i)
            flow_df[mask].to_excel(writer, sheet_name=f"Flow_C{i + 1}", index=False)
            rain_df[mask].to_excel(writer, sheet_name=f"Rain_C{i + 1}", index=False)

    # Visualization
    plt.figure(figsize=(8, 6))
    for i in range(CFG["n_clusters"]):
        plt.subplot(CFG["n_clusters"], 1, i + 1)
        plt.plot(weighted_data[labels == i].T, "k-", alpha=0.1)
        plt.plot(weighted_data[km.medoid_indices_[i]], "r-", label=f"Center {i + 1}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CFG['save_dir']}cluster_centers.png")
    print(f"Pipeline complete. Files saved in {CFG['save_dir']}")


if __name__ == "__main__":
    run_pipeline()