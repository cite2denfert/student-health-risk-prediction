"""
평가 리포트 생성 (혼동행렬, 피처 중요도)

    python -m src.evaluate
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src import config
from src.data import load_train, split_train_valid
from src.pipeline import prepare_features


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def plot_feature_importance(pipe, X, save_path, top_n=20):
    """
    calibrated 모델은 base_estimator 파이프라인 리스트를 감싸고 있으므로,
    permutation_importance 로 원본 컬럼 기준 중요도를 계산합니다
    (Top 20 Feature Importance).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipe, X, pipe.predict(X), n_repeats=3,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=X.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    importances.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_n} Feature Importance (permutation)")
    ax.set_xlabel("Mean decrease in score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return importances


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 가 없습니다. 먼저 `python -m src.train` 을 실행하세요.")

    calibrated = joblib.load(model_path)

    df = load_train()
    _, valid_df = split_train_valid(df)
    y_valid = valid_df[config.TARGET_COL]
    X_valid = prepare_features(valid_df)

    pred = calibrated.predict(X_valid)
    labels = sorted(y_valid.unique())

    cm = plot_confusion_matrix(y_valid, pred, labels, config.FIGURES_DIR / "confusion_matrix.png")
    print("Confusion matrix (row-normalized):")
    print(pd.DataFrame(cm, index=labels, columns=labels).round(3))

    # 피처 중요도는 오래 걸릴 수 있어 표본으로 계산
    sample = X_valid.sample(min(20_000, len(X_valid)), random_state=config.RANDOM_STATE)
    importances = plot_feature_importance(calibrated, sample, config.FIGURES_DIR / "feature_importance.png")
    print("\nTop feature importances:")
    print(importances.head(10))

    print(f"\n그래프 저장 위치: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
