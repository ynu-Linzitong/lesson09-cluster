"""第9课：使用课堂展示的300位客户数据完成K-Means聚类与自主实践。"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mall_customers_yuan.csv"
FEATURES = ["income", "balance", "product_count"]
EXTENDED_FEATURES = FEATURES + ["age"]
FEATURE_NAMES = {
    "income": "年收入",
    "balance": "账户余额",
    "product_count": "持有产品数",
    "age": "年龄",
}
K = 4
ELBOW_K = 3
SEEDS = [42, 7, 2026]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def business_names(profile: pd.DataFrame) -> dict[int, str]:
    """按收入和产品持有特征命名，避免把名称绑定到随机簇编号。"""
    highest_income = int(profile["income"].idxmax())
    lowest_income = int(profile["income"].idxmin())
    remaining = [int(c) for c in profile.index if int(c) not in {highest_income, lowest_income}]
    remaining.sort(key=lambda c: profile.loc[c, "product_count"], reverse=True)
    names = {
        highest_income: "高净值富裕型",
        lowest_income: "年轻起步培育型",
    }
    if len(remaining) == 2:
        names[remaining[0]] = "多产品活跃成长型"
        names[remaining[1]] = "中产稳健维护型"
    return names


def fit_kmeans(data: pd.DataFrame, features: list[str], seed: int = 42) -> tuple[KMeans, np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(data[features])
    model = KMeans(n_clusters=K, random_state=seed, n_init=50)
    labels = model.fit_predict(scaled)
    return model, labels, scaled


def make_profile(data: pd.DataFrame, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    clustered = data.copy()
    clustered["cluster"] = labels
    profile_features = list(dict.fromkeys([*features, "age"]))
    profile = clustered.groupby("cluster")[profile_features].mean()
    names = business_names(profile)
    profile.insert(0, "customer_count", clustered.groupby("cluster").size())
    profile.insert(0, "business_name", pd.Series(names))
    return profile


def save_elbow_and_silhouette(scaled: np.ndarray, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    ks = list(range(1, 10))
    inertias = []
    silhouettes = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(scaled, labels)) if k >= 2 else np.nan)

    elbow = pd.DataFrame({"k": ks, "SSE": inertias})
    elbow["SSE_drop"] = elbow["SSE"].shift(1) - elbow["SSE"]
    elbow["SSE_drop_percent"] = elbow["SSE_drop"] / elbow["SSE"].shift(1) * 100
    elbow.to_csv(output_dir / "elbow_scores.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ks, inertias, marker="o", linewidth=2, color="#2563EB")
    ax.scatter([ELBOW_K], [inertias[ELBOW_K - 1]], s=170, color="#F97316", edgecolor="white", zorder=5)
    ax.scatter([K], [inertias[K - 1]], s=170, color="#DC2626", edgecolor="white", zorder=5)
    ax.annotate(
        f"主要肘点 k={ELBOW_K}",
        xy=(ELBOW_K, inertias[ELBOW_K - 1]),
        xytext=(-85, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C2410C"},
        fontsize=11,
    )
    ax.annotate(
        f"作业四簇方案 k={K}",
        xy=(K, inertias[K - 1]),
        xytext=(35, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#B91C1C"},
        fontsize=11,
    )
    ax.set(title="肘部法则与四簇客户细分方案", xlabel="聚类数 k", ylabel="组内误差平方和 SSE", xticks=ks)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "elbow.png", dpi=180)
    plt.close(fig)

    silhouette = pd.DataFrame({"k": ks[1:], "silhouette_score": silhouettes[1:]})
    silhouette.to_csv(output_dir / "silhouette_scores.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ks[1:], silhouettes[1:], marker="o", linewidth=2, color="#059669")
    ax.axvline(K, linestyle="--", color="#DC2626", label=f"作业方案 k={K}")
    best_k = int(silhouette.loc[silhouette["silhouette_score"].idxmax(), "k"])
    ax.axvline(best_k, linestyle=":", color="#059669", label=f"轮廓系数最高 k={best_k}")
    ax.set(title="轮廓系数（独立研究）", xlabel="聚类数 k", ylabel="Silhouette score", xticks=ks[1:])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "silhouette.png", dpi=180)
    plt.close(fig)
    return elbow, silhouette, inertias


def save_profile_plot(profile: pd.DataFrame, data: pd.DataFrame, output_dir: Path) -> None:
    z_profile = profile[FEATURES].copy()
    for feature in FEATURES:
        z_profile[feature] = (z_profile[feature] - data[feature].mean()) / data[feature].std()
    x = np.arange(len(FEATURES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#2563EB", "#F97316", "#059669", "#DC2626"]
    for index, (cluster, row) in enumerate(z_profile.iterrows()):
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            row.values,
            width,
            label=f"簇{cluster}：{profile.loc[cluster, 'business_name']}",
            color=colors[index % len(colors)],
        )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set(
        title="四类客户画像（各簇均值相对总体的标准差位置）",
        ylabel="标准化后的簇均值（总体均值=0）",
        xticks=x,
        xticklabels=[FEATURE_NAMES[feature] for feature in FEATURES],
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "customer_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scatter(data: pd.DataFrame, labels: np.ndarray, profile: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563EB", "#F97316", "#059669", "#DC2626"]
    for cluster in range(K):
        mask = labels == cluster
        ax.scatter(
            data.loc[mask, "income"],
            data.loc[mask, "balance"],
            s=34,
            alpha=0.75,
            color=colors[cluster],
            label=f"簇{cluster}：{profile.loc[cluster, 'business_name']}",
        )
    ax.set(title="客户分群二维投影（自主实践）", xlabel="年收入 income（万元）", ylabel="账户余额 balance（元）")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_scatter.png", dpi=180)
    plt.close(fig)


def stability_check(data: pd.DataFrame, baseline_labels: np.ndarray, baseline_model: KMeans, output_dir: Path) -> pd.DataFrame:
    models = {42: (baseline_model, baseline_labels)}
    for seed in SEEDS[1:]:
        models[seed] = fit_kmeans(data, FEATURES, seed)[:2]

    count_rows = []
    for seed, (_, labels) in models.items():
        counts = pd.Series(labels).value_counts().sort_index()
        count_rows.append({"random_state": seed, **{f"cluster_{i}_count": int(counts.get(i, 0)) for i in range(K)}})
    pd.DataFrame(count_rows).to_csv(output_dir / "stability_cluster_counts.csv", index=False, encoding="utf-8-sig")

    rows = []
    for first, second in combinations(SEEDS, 2):
        first_model, first_labels = models[first]
        second_model, second_labels = models[second]
        distance = np.linalg.norm(first_model.cluster_centers_[:, None, :] - second_model.cluster_centers_[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(distance)
        center_differences = distance[row_ind, col_ind]
        rows.append(
            {
                "seed_pair": f"{first}_vs_{second}",
                "adjusted_rand_index": adjusted_rand_score(first_labels, second_labels),
                "max_aligned_center_distance": float(center_differences.max()),
                "mean_aligned_center_distance": float(center_differences.mean()),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(output_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")
    return stability


def ring_experiment(output_dir: Path) -> dict[str, float | int]:
    # 这里生成的是算法玩具数据，不是客户信息，仅用于验证非凸簇。
    points, true_labels = make_circles(n_samples=400, factor=0.5, noise=0.05, random_state=42)
    kmeans_labels = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(points)
    dbscan_labels = DBSCAN(eps=0.18, min_samples=5).fit_predict(points)
    metrics = {
        "kmeans_ari": float(adjusted_rand_score(true_labels, kmeans_labels)),
        "dbscan_ari": float(adjusted_rand_score(true_labels, dbscan_labels)),
        "dbscan_non_noise_clusters": int(len(set(dbscan_labels) - {-1})),
        "dbscan_noise_points": int(np.sum(dbscan_labels == -1)),
    }
    with (output_dir / "ring_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, labels, title in zip(
        axes,
        [kmeans_labels, dbscan_labels],
        ["K-Means：直线边界切开环形", "DBSCAN：按密度识别两个环"],
    ):
        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", s=18)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.suptitle("环形玩具数据实验（非客户数据）")
    fig.tight_layout()
    fig.savefig(output_dir / "ring_dbscan_comparison.png", dpi=180)
    plt.close(fig)
    return metrics


def main(data_path: Path | None = None) -> None:
    selected_path = data_path or DATA_PATH
    if not selected_path.exists():
        raise FileNotFoundError(f"找不到课堂展示数据：{selected_path}")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(selected_path)
    required = ["customer_id", *EXTENDED_FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段：{missing}")
    if data[EXTENDED_FEATURES].isna().any().any():
        raise ValueError("特征列存在缺失值，请先完成缺失值处理。")

    model, labels, scaled = fit_kmeans(data, FEATURES)
    profile = make_profile(data, labels, FEATURES)
    profile.to_csv(output_dir / "customer_profiles.csv", encoding="utf-8-sig")
    counts = profile[["business_name", "customer_count"]].copy()
    counts.index.name = "cluster"
    counts.to_csv(output_dir / "cluster_counts.csv", encoding="utf-8-sig")
    clustered = data.copy()
    clustered["cluster"] = labels
    clustered["business_name"] = clustered["cluster"].map(profile["business_name"])
    clustered.to_csv(output_dir / "customer_clusters.csv", index=False, encoding="utf-8-sig")

    _, silhouette, inertias = save_elbow_and_silhouette(scaled, output_dir)
    save_profile_plot(profile, data, output_dir)
    save_scatter(data, labels, profile, output_dir)

    _, extended_labels, _ = fit_kmeans(data, EXTENDED_FEATURES)
    extended_profile = make_profile(data, extended_labels, EXTENDED_FEATURES)
    extended_profile.to_csv(output_dir / "extended_customer_profiles.csv", encoding="utf-8-sig")
    extended_ari = float(adjusted_rand_score(labels, extended_labels))

    stability = stability_check(data, labels, model, output_dir)

    unscaled_model = KMeans(n_clusters=K, random_state=42, n_init=50)
    unscaled_labels = unscaled_model.fit_predict(data[FEATURES])
    unscaled_profile = make_profile(data, unscaled_labels, FEATURES)
    unscaled_profile.to_csv(output_dir / "unscaled_customer_profiles.csv", encoding="utf-8-sig")
    unscaled_ari = float(adjusted_rand_score(labels, unscaled_labels))

    ring_metrics = ring_experiment(output_dir)
    selected_silhouette = float(silhouette.loc[silhouette["k"] == K, "silhouette_score"].iloc[0])
    best_silhouette_k = int(silhouette.loc[silhouette["silhouette_score"].idxmax(), "k"])
    try:
        source_name = str(selected_path.relative_to(ROOT))
    except ValueError:
        source_name = str(selected_path)
    metrics = {
        "data_source": source_name,
        "dataset_note": "课堂展示数据，不是教师指南中的bank_customers.csv；未生成或修改客户信息。",
        "sample_count": int(len(data)),
        "base_features": FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "k": K,
        "elbow_k": ELBOW_K,
        "cluster_counts": {str(int(cluster)): int(row["customer_count"]) for cluster, row in profile.iterrows()},
        "business_names": {str(int(cluster)): str(row["business_name"]) for cluster, row in profile.iterrows()},
        "sse_by_k": {str(k): float(sse) for k, sse in enumerate(inertias, start=1)},
        "selected_silhouette": selected_silhouette,
        "best_silhouette_k": best_silhouette_k,
        "best_silhouette_score": float(silhouette["silhouette_score"].max()),
        "extended_features_ari": extended_ari,
        "extended_cluster_counts": {str(int(cluster)): int(row["customer_count"]) for cluster, row in extended_profile.iterrows()},
        "extended_business_names": {str(int(cluster)): str(row["business_name"]) for cluster, row in extended_profile.iterrows()},
        "minimum_stability_ari": float(stability["adjusted_rand_index"].min()),
        "maximum_aligned_center_distance": float(stability["max_aligned_center_distance"].max()),
        "unscaled_vs_scaled_ari": unscaled_ari,
        "unscaled_cluster_counts": {str(int(cluster)): int(row["customer_count"]) for cluster, row in unscaled_profile.iterrows()},
        "feature_standard_deviations": {feature: float(data[feature].std()) for feature in FEATURES},
        "ring_experiment": ring_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"数据文件：{metrics['data_source']}")
    print(f"样本数：{len(data)}；聚类特征：{FEATURES}；k={K}")
    print("四簇人数：", metrics["cluster_counts"])
    print("四个业务命名：", metrics["business_names"])
    print(f"主要肘点：k={ELBOW_K}；作业采用四簇方案：k={K}")
    print(f"k={K} 轮廓系数：{selected_silhouette:.4f}；最高轮廓系数出现在 k={best_silhouette_k}")
    print(f"三种子最低 ARI：{metrics['minimum_stability_ari']:.4f}")
    print(f"加入年龄前后 ARI：{extended_ari:.4f}")
    print(f"未标准化与标准化结果 ARI：{unscaled_ari:.4f}")
    print(f"结果已保存到：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="可选：指定同字段CSV；默认使用data/mall_customers_yuan.csv")
    arguments = parser.parse_args()
    main(arguments.data)
