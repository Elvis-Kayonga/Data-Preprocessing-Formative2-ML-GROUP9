from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "outputs" / "plots"

SOCIAL_FILE = RAW_DIR / "customer_social_profiles.csv"
TRANSACTIONS_FILE = RAW_DIR / "customer_transactions.csv"


def load_data():
    social = pd.read_csv(SOCIAL_FILE)
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    return social, transactions


def clean_social_profiles(social):
    social = social.copy()
    social["customer_id"] = social["customer_id_new"].str.replace("A", "", regex=False).astype(int)
    social = social.drop_duplicates()
    for col in ["social_media_platform", "review_sentiment"]:
        if social[col].isna().any():
            social[col] = social[col].fillna(social[col].mode().iloc[0])
    for col in ["engagement_score", "purchase_interest_score"]:
        if social[col].isna().any():
            social[col] = social[col].fillna(social[col].median())
    return social


def aggregate_social_profiles(social):
    grouped = social.groupby("customer_id", as_index=False).agg(
        engagement_mean=("engagement_score", "mean"),
        engagement_max=("engagement_score", "max"),
        engagement_min=("engagement_score", "min"),
        purchase_interest_mean=("purchase_interest_score", "mean"),
        social_record_count=("customer_id_new", "count"),
        social_media_platform_mode=("social_media_platform", lambda s: s.mode().iloc[0]),
        review_sentiment_mode=("review_sentiment", lambda s: s.mode().iloc[0]),
    )
    return grouped


def clean_transactions(transactions):
    transactions = transactions.copy()
    transactions["customer_id"] = transactions["customer_id_legacy"].astype(int)
    transactions = transactions.drop_duplicates()
    transactions["customer_rating"] = pd.to_numeric(transactions["customer_rating"], errors="coerce")
    transactions["customer_rating"] = transactions["customer_rating"].fillna(transactions["customer_rating"].median())
    transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"], errors="coerce")
    transactions["purchase_month"] = transactions["purchase_date"].dt.month
    transactions["purchase_day_of_week"] = transactions["purchase_date"].dt.dayofweek
    if transactions["purchase_month"].isna().any():
        transactions["purchase_month"] = transactions["purchase_month"].fillna(
            int(transactions["purchase_month"].median())
        )
    if transactions["purchase_day_of_week"].isna().any():
        transactions["purchase_day_of_week"] = transactions["purchase_day_of_week"].fillna(
            int(transactions["purchase_day_of_week"].median())
        )
    return transactions


def merge_datasets(social_agg, transactions):
    merged = transactions.merge(social_agg, how="left", on="customer_id")
    merged["engagement_mean"] = merged["engagement_mean"].fillna(merged["engagement_mean"].median())
    merged["engagement_max"] = merged["engagement_max"].fillna(merged["engagement_max"].median())
    merged["engagement_min"] = merged["engagement_min"].fillna(merged["engagement_min"].median())
    merged["purchase_interest_mean"] = merged["purchase_interest_mean"].fillna(
        merged["purchase_interest_mean"].median()
    )
    merged["social_record_count"] = merged["social_record_count"].fillna(0)
    merged["social_media_platform_mode"] = merged["social_media_platform_mode"].fillna("Unknown")
    merged["review_sentiment_mode"] = merged["review_sentiment_mode"].fillna("Unknown")
    return merged


def generate_required_plots(merged):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "purchase_amount", "customer_rating", "engagement_mean",
        "purchase_interest_mean", "social_record_count",
    ]

    # Plot 01 — 4-panel distribution histograms.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col, color, title in zip(
        axes.flat,
        ["purchase_amount", "engagement_mean", "customer_rating", "purchase_interest_mean"],
        ["#007f5f", "#2b9348", "#40916c", "#52b788"],
        [
            "Distribution of Purchase Amount",
            "Distribution of Avg Engagement Score",
            "Distribution of Customer Rating",
            "Distribution of Purchase Interest Score",
        ],
    ):
        sns.histplot(merged[col], kde=True, ax=ax, color=color, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Feature Distributions — Merged Customer Dataset", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 02 — 4-panel outlier boxplots.
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    box_cols = ["purchase_amount", "customer_rating", "engagement_mean", "purchase_interest_mean"]
    box_colors = ["#80b918", "#55a630", "#2d6a4f", "#74c69d"]
    for ax, col, color in zip(axes, box_cols, box_colors):
        sns.boxplot(y=merged[col], ax=ax, color=color, width=0.5,
                    flierprops={"marker": "o", "markersize": 4})
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Outlier Detection — Boxplot Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_outliers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 03 — Correlation heatmap.
    corr = merged[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.6, ax=ax,
        annot_kws={"size": 11}, vmin=-1, vmax=1, center=0, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap of Engineered Numeric Features", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 05 — Target class distribution (bar + pie).
    if "product_category" in merged.columns:
        cat_counts = merged["product_category"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        bar_colors = sns.color_palette("YlGn", len(cat_counts))
        axes[0].bar(cat_counts.index, cat_counts.values, color=bar_colors, edgecolor="white", linewidth=0.8)
        axes[0].set_title("Product Category — Transaction Counts", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Product Category")
        axes[0].set_ylabel("Number of Transactions")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", alpha=0.3)
        for bar, val in zip(axes[0].patches, cat_counts.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9,
            )
        pie_colors = sns.color_palette("YlGn", len(cat_counts))
        _, _, autotexts = axes[1].pie(
            cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
            colors=pie_colors, startangle=140, pctdistance=0.85,
        )
        for text in autotexts:
            text.set_fontsize(9)
        axes[1].set_title("Product Category — Proportions", fontsize=12, fontweight="bold")
        fig.suptitle("Target Variable Distribution: Product Category", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "05_category_distribution.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot 06 — Purchase amount per category.
    if "product_category" in merged.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        order = merged.groupby("product_category")["purchase_amount"].median().sort_values(ascending=False).index
        sns.boxplot(
            data=merged, x="product_category", y="purchase_amount",
            hue="product_category", order=order, palette="YlGn",
            ax=ax, width=0.55, legend=False,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        ax.set_title("Purchase Amount Distribution by Product Category", fontsize=13, fontweight="bold")
        ax.set_xlabel("Product Category", fontsize=11)
        ax.set_ylabel("Purchase Amount ($)", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "06_category_purchase_boxplot.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot 07 — Social platform & sentiment overview.
    if "social_media_platform_mode" in merged.columns and "review_sentiment_mode" in merged.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plat_counts = merged["social_media_platform_mode"].value_counts()
        axes[0].bar(plat_counts.index, plat_counts.values,
                    color=sns.color_palette("YlGn", len(plat_counts)), edgecolor="white")
        axes[0].set_title("Dominant Social Media Platform per Customer", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Platform")
        axes[0].set_ylabel("Number of Customers")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=0.3)
        sent_counts = merged["review_sentiment_mode"].value_counts()
        sentiment_colors = {"Positive": "#40916c", "Neutral": "#74c69d", "Negative": "#d62828", "Unknown": "#adb5bd"}
        s_colors = [sentiment_colors.get(s, "#999") for s in sent_counts.index]
        axes[1].bar(sent_counts.index, sent_counts.values, color=s_colors, edgecolor="white")
        axes[1].set_title("Review Sentiment Distribution", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Sentiment")
        axes[1].set_ylabel("Number of Customers")
        axes[1].grid(axis="y", alpha=0.3)
        fig.suptitle("Social Profile Overview", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "07_social_overview.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    social_raw, transactions_raw = load_data()
    social_clean = clean_social_profiles(social_raw)
    social_agg = aggregate_social_profiles(social_clean)
    transactions_clean = clean_transactions(transactions_raw)
    merged = merge_datasets(social_agg, transactions_clean)
    social_clean.to_csv(PROCESSED_DIR / "social_profiles_clean.csv", index=False)
    social_agg.to_csv(PROCESSED_DIR / "social_profiles_aggregated.csv", index=False)
    transactions_clean.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
    merged.to_csv(PROCESSED_DIR / "merged_customer_dataset.csv", index=False)
    generate_required_plots(merged)
    print("Saved cleaned/merged data and required plots.")


if __name__ == "__main__":
    main()
