from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
MERGED_FILE = ROOT / "data" / "processed" / "merged_customer_dataset.csv"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"

NUMERIC_FEATURES = [
    "purchase_amount", "customer_rating", "purchase_month", "purchase_day_of_week",
    "engagement_mean", "engagement_max", "engagement_min",
    "purchase_interest_mean", "social_record_count",
]
CATEGORICAL_FEATURES = ["social_media_platform_mode", "review_sentiment_mode"]


def load_merged_data():
    return pd.read_csv(MERGED_FILE)


def build_features_and_target(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["product_category"]
    return X, y


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


def build_models(n_classes):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_split=4,
            min_samples_leaf=2, random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=5000, solver="lbfgs", random_state=42,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            num_class=n_classes, random_state=42,
        )
    return models


def build_pipeline(model):
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "log_loss": round(log_loss(y_test, y_proba, labels=pipeline.classes_), 4),
    }


def save_model_comparison_plot(comparison):
    """Grouped bar chart with value annotations comparing all models."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(comparison.keys())
    display_names = [n.replace("_", " ").title() for n in model_names]
    accuracy_vals = [comparison[n]["accuracy"] for n in model_names]
    f1_vals = [comparison[n]["f1_weighted"] for n in model_names]
    loss_vals = [comparison[n]["log_loss"] for n in model_names]
    x = np.arange(len(model_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for vals, offset, label, color in [
        (accuracy_vals, -width, "Accuracy", "#2d6a4f"),
        (f1_vals, 0, "F1-Score (weighted)", "#52b788"),
        (loss_vals, width, "Log Loss", "#d62828"),
    ]:
        bars = ax.bar(x + offset, vals, width=width, label=label, color=color, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5,
            )
    ax.set_title("Model Comparison: Accuracy / F1-Score / Log Loss", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_model_comparison_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(pipeline, X_test, y_test, model_name):
    """Annotated confusion matrix heatmap for the chosen model."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGn",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax, annot_kws={"size": 10},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix — {model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(pipeline, feature_names):
    """Horizontal bar chart of the top-15 feature importances (RF only)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ohe = pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_names = []
    all_names = NUMERIC_FEATURES + cat_names
    importances = model.feature_importances_
    n = min(len(importances), len(all_names))
    importances = importances[:n]
    all_names = all_names[:n]
    sorted_idx = importances.argsort()[::-1]
    top_n = min(15, n)
    top_idx = sorted_idx[:top_n]
    top_names = [all_names[i].replace("_", " ").title() for i in top_idx]
    top_vals = importances[top_idx]
    palette = sns.color_palette("YlGn_r", top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_vals[::-1], color=palette[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Top Feature Importances — Random Forest", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_merged_data()
    X, y = build_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = build_models(n_classes=y.nunique())
    comparison, trained_pipelines, skipped_models = {}, {}, []
    if not XGBOOST_AVAILABLE:
        skipped_models.append("xgboost (not importable in active environment)")
    for model_name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        comparison[model_name] = evaluate_pipeline(pipeline, X_test, y_test)
        trained_pipelines[model_name] = pipeline
        print(f"  [{model_name}] {comparison[model_name]}")
    best_model_name = min(comparison, key=lambda name: comparison[name]["log_loss"])
    best_pipeline = trained_pipelines[best_model_name]
    summary = {
        "best_model": best_model_name,
        "best_metrics": comparison[best_model_name],
        "all_model_metrics": comparison,
        "skipped_models": skipped_models,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_classes": int(y.nunique()),
        "target_classes": sorted(y.unique().tolist()),
    }
    joblib.dump(best_pipeline, MODELS_DIR / "product_recommendation_model.joblib")
    with open(MODELS_DIR / "product_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(MODELS_DIR / "product_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    save_model_comparison_plot(comparison)
    save_confusion_matrix_plot(best_pipeline, X_test, y_test, best_model_name)
    save_feature_importance_plot(best_pipeline, list(X.columns))
    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
