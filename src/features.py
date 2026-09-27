"""
파생변수 생성
------------
아래 파생변수들을 생성합니다.

    fatigue_index       = stress + screen_time - sleep   → 피로 누적 정도
    activity_index       = exercise + steps               → 신체 활동 수준
    health_habit_score    = sleep + hydration + exercise    → 건강 생활 습관 점수
    bmi_step_ratio        = BMI / steps                     → 체중 대비 활동량
    sleep_x_quality       = sleep × sleep_quality            → 수면의 양과 질 반영

Kaggle Playground S6E7 원본 데이터에는 `screen_time` / `study_time` 컬럼이
없을 수 있습니다. 해당 컬럼이 없으면
자동으로 생략하고 경고만 출력하므로, 스크립트가 죽지 않고 실행됩니다.
실제 컬럼을 확인한 뒤 이 파일의 수식을 데이터에 맞게 다듬어 쓰세요.

Ablation (HistGradientBoosting, valid BA): 파생변수를 빼면 0.9424 → 0.9093 (-0.033).
class_weight를 빼면 0.8663 (-0.076)으로, 점수에 가장 큰 영향을 주는 건 불균형 처리입니다.
"""
import warnings

import numpy as np
import pandas as pd

# 순서형 범주 → 점수 매핑 (데이터에 맞게 조정 가능)
STRESS_ORDER = {"low": 0, "medium": 1, "moderate": 1, "high": 2}
SLEEP_QUALITY_ORDER = {"poor": 0, "fair": 1, "average": 1, "good": 2, "excellent": 3}


def _ordinal_score(series: pd.Series, mapping: dict) -> pd.Series:
    """object dtype이면 매핑으로 점수화하고, 이미 수치형이면 그대로 반환."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    normalized = series.astype(str).str.strip().str.lower()
    scored = normalized.map(mapping)
    if scored.isna().any():
        unknown = sorted(set(normalized.dropna()) - set(mapping))
        if unknown:
            warnings.warn(f"알 수 없는 범주값 발견, NaN 처리됨: {unknown}")
    return scored


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """3축 원본 변수로부터 파생변수를 계산해 새 컬럼으로 추가합니다."""
    out = df.copy()

    def has(*cols: str) -> bool:
        missing = [c for c in cols if c not in out.columns]
        if missing:
            warnings.warn(f"컬럼 없음 {missing} → 관련 파생변수 생략")
        return not missing

    # 행 단위 결측치 개수 (EDA에서 소수 클래스 판별에 쓰인 num__n_missing)
    out["n_missing"] = df.isna().sum(axis=1)

    if has("stress_level", "sleep_duration"):
        stress_score = _ordinal_score(out["stress_level"], STRESS_ORDER)
        out["fatigue_index"] = stress_score - out["sleep_duration"]
        if "screen_time" in out.columns:
            out["fatigue_index"] = out["fatigue_index"] + out["screen_time"]

    if has("exercise_duration", "step_count"):
        out["activity_index"] = out["exercise_duration"] + out["step_count"] / 1000.0

    if has("sleep_duration", "water_intake", "exercise_duration"):
        out["health_habit_score"] = (
            out["sleep_duration"] + out["water_intake"] + out["exercise_duration"]
        )

    if has("bmi", "step_count"):
        out["bmi_step_ratio"] = out["bmi"] / (out["step_count"] + 1)

    if has("sleep_duration", "sleep_quality"):
        quality_score = _ordinal_score(out["sleep_quality"], SLEEP_QUALITY_ORDER)
        out["sleep_x_quality"] = out["sleep_duration"] * quality_score

    if has("heart_rate", "bmi"):
        out["hr_per_bmi"] = out["heart_rate"] / out["bmi"].replace(0, np.nan)

    if has("calorie_expenditure", "step_count"):
        out["cal_per_step"] = out["calorie_expenditure"] / (out["step_count"] + 1)

    if has("calorie_expenditure", "exercise_duration"):
        out["cal_per_min"] = out["calorie_expenditure"] / (out["exercise_duration"] + 1)

    return out
