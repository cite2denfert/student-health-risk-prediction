"""
학습 파이프라인 (7단계)

    ① 데이터 준비
    ② Train / Valid 분리 (stratify)
    ③ ColumnTransformer (수치형: 중앙값 대체+표준화 / 범주형: 최빈값 대체+원핫)
    ④ 모델 학습 (Logistic → RF → CatBoost → XGBoost → HistGB → ...)
    ⑤ Balanced Accuracy 평가
    ⑥ 최종 모델 선정 + 확률 보정(Calibration)
    ⑦ (predict.py 에서) Kaggle 제출 파일 생성

사용법:
    python -m src.train                     # 전체 데이터로 학습
    python -m src.train --sample 150000      # 모델 비교 단계 표본 크기 조절
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

from src import config
from src.data import load_train, split_train_valid
from src.models import PriorAdjustedClassifier, get_candidate_models
from src.pipeline import build_preprocessor, prepare_features


def bake_off(X_train, y_train, X_valid, y_valid, sample_size: int | None) -> pd.DataFrame:
    """후보 모델 7종을 동일한 파이프라인으로 학습해 Balanced Accuracy로 비교."""
    if sample_size and sample_size < len(X_train):
        X_bo = X_train.sample(sample_size, random_state=config.RANDOM_STATE)
        y_bo = y_train.loc[X_bo.index]
    else:
        X_bo, y_bo = X_train, y_train

    results = []
    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", model)])
        start = time.time()
        pipe.fit(X_bo, y_bo)
        elapsed = time.time() - start

        pred = pipe.predict(X_valid)
        ba = balanced_accuracy_score(y_valid, pred)
        acc = accuracy_score(y_valid, pred)
        results.append({"model": name, "balanced_accuracy": ba, "accuracy": acc, "train_seconds": elapsed})
        print(f"  {name:<20s} BA={ba:.4f}  Acc={acc:.4f}  time={elapsed:.1f}s")

    return pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=150_000,
                         help="모델 비교(bake-off) 단계에서 사용할 학습 표본 크기. "
                              "0 이면 전체 데이터를 사용합니다.")
    args = parser.parse_args()
    sample_size = args.sample if args.sample > 0 else None

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("① 데이터 준비")
    df = load_train()

    print("② Train / Valid 분리 (stratify)")
    train_df, valid_df = split_train_valid(df)
    y_train, y_valid = train_df[config.TARGET_COL], valid_df[config.TARGET_COL]

    print("③ 파생변수 생성 + 전처리 파이프라인 구성")
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    print(f"④~⑤ 모델 비교 (bake-off, 표본={sample_size or 'ALL'})")
    leaderboard = bake_off(X_train, y_train, X_valid, y_valid, sample_size)
    leaderboard.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    print("\n[리더보드]")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["model"]
    print(f"\n⑥ 최종 모델 선정: {best_name} (전체 학습 데이터로 재학습)")

    best_model = get_candidate_models()[best_name]
    final_pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", best_model)])
    final_pipe.fit(X_train, y_train)

    pred_before = final_pipe.predict(X_valid)
    ba_before = balanced_accuracy_score(y_valid, pred_before)
    acc_before = accuracy_score(y_valid, pred_before)
    print(f"  보정 전 Valid — BA={ba_before:.4f}  Acc={acc_before:.4f}")

    print("  확률 보정 (Calibration, sigmoid, cv=3)")
    calibrated = CalibratedClassifierCV(final_pipe, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    pred_naive = calibrated.predict(X_valid)
    ba_naive = balanced_accuracy_score(y_valid, pred_naive)
    acc_naive = accuracy_score(y_valid, pred_naive)
    print(f"  보정 후(단순 argmax) Valid — BA={ba_naive:.4f}  Acc={acc_naive:.4f}")
    print("  (참고) 단순 argmax는 보정이 클래스 사전확률로 확률을 되돌리면서")
    print("         소수 클래스 재현율이 깎여 BA가 오히려 하락할 수 있음 → 사전확률 보정 적용")

    class_prior = y_train.value_counts(normalize=True).to_dict()
    final_model = PriorAdjustedClassifier(calibrated, class_prior=class_prior).fit(X_train, y_train)

    pred_after = final_model.predict(X_valid)
    ba_after = balanced_accuracy_score(y_valid, pred_after)
    acc_after = accuracy_score(y_valid, pred_after)
    print(f"  보정 후(사전확률 보정) Valid — BA={ba_after:.4f}  Acc={acc_after:.4f}")

    joblib.dump(final_model, config.MODELS_DIR / "final_model.joblib")
    joblib.dump(final_pipe, config.MODELS_DIR / "final_model_uncalibrated.joblib")

    metrics = {
        "best_model": best_name,
        "balanced_accuracy_before_calibration": ba_before,
        "accuracy_before_calibration": acc_before,
        "balanced_accuracy_after_calibration_naive_argmax": ba_naive,
        "accuracy_after_calibration_naive_argmax": acc_naive,
        "balanced_accuracy_after_calibration_prior_adjusted": ba_after,
        "accuracy_after_calibration_prior_adjusted": acc_after,
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "bake_off_sample_size": sample_size or len(X_train),
    }
    with open(config.METRICS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {config.MODELS_DIR / 'final_model.joblib'}")
    print(f"메트릭 저장: {config.METRICS_DIR / 'final_metrics.json'}")


if __name__ == "__main__":
    main()
