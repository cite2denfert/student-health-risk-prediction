"""
EDA 리포트 생성 (타깃 분포, 결측 구조, 클래스별 피로도 + 통계검정)

    python -m src.eda
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from src import config
from src.data import load_train
from src.features import add_derived_features


def plot_target_and_missing(df: pd.DataFrame, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df[config.TARGET_COL].value_counts().reindex(config.TARGET_CLASSES).plot.bar(
        ax=axes[0], color=["#c0392b", "#e67e22", "#27ae60"]
    )
    axes[0].set_title("Target distribution (imbalanced)")
    axes[0].set_xlabel(config.TARGET_COL)

    na_ratio = df.isna().mean().sort_values(ascending=False).head(10) * 100
    na_ratio.sort_values().plot.barh(ax=axes[1], color="#8e44ad")
    axes[1].set_title("Missing values (top 10, %)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fatigue_by_class(df: pd.DataFrame, save_path):
    enriched = add_derived_features(df)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cls, color in zip(config.TARGET_CLASSES, ["#3498db", "#e67e22", "#27ae60"]):
        subset = enriched.loc[enriched[config.TARGET_COL] == cls, "fatigue_index"].dropna()
        ax.hist(subset, bins=40, density=True, alpha=0.5, label=cls, color=color)
    ax.set_title("fatigue_index by class")
    ax.set_xlabel("fatigue_index")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return enriched


def anova_effect_sizes(enriched: pd.DataFrame) -> pd.DataFrame:
    """통계검정(ANOVA, eta-squared) 재현: 파생변수별 클래스 간 유의성."""
    rows = []
    for col in config.DERIVED_FEATURE_NAMES:
        if col not in enriched.columns:
            continue
        groups = [
            enriched.loc[enriched[config.TARGET_COL] == cls, col].dropna()
            for cls in config.TARGET_CLASSES
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_value = stats.f_oneway(*groups)

        # eta-squared = SS_between / SS_total
        all_vals = pd.concat(groups)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((all_vals - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

        rows.append({"feature": col, "f_stat": f_stat, "p_value": p_value, "eta_squared": eta_sq})

    result = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    return result


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train()

    plot_target_and_missing(df, config.FIGURES_DIR / "target_and_missing.png")
    enriched = plot_fatigue_by_class(df, config.FIGURES_DIR / "fatigue_index_by_class.png")

    effect_sizes = anova_effect_sizes(enriched)
    effect_sizes.to_csv(config.METRICS_DIR / "anova_effect_sizes.csv", index=False)

    print("타깃 분포:")
    print(df[config.TARGET_COL].value_counts(normalize=True).round(4) * 100)
    print("\n파생변수 ANOVA (클래스 간 차이 유의성, eta-squared로 정렬):")
    print(effect_sizes.to_string(index=False))
    print(f"\n그래프/표 저장 위치: {config.FIGURES_DIR}, {config.METRICS_DIR}")


if __name__ == "__main__":
    main()
