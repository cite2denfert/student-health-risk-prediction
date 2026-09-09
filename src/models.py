"""후보 모델 정의 (머신러닝 모델 비교 — 7종 오디션)"""
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost의 sklearn API는 0..K-1 정수 라벨을 요구합니다.
    문자열 타깃('at-risk' 등)을 그대로 다른 모델들과 동일하게 다루기 위해
    내부적으로 LabelEncoder를 적용했다가 predict 시 원래 문자열로 복원하는
    얇은 래퍼입니다.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_enc)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        return self.label_encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class PriorAdjustedClassifier(BaseEstimator, ClassifierMixin):
    """
    확률 보정(Calibration) 이후 argmax로 바로 분류하면, sigmoid 보정이
    확률을 실제 클래스 비율(85.9% / 8.4% / 5.8%)쪽으로 되돌리면서
    class_weight="balanced" 로 학습 때 얻은 소수 클래스 재현율 이득이
    상쇄되어 버립니다 (실험적으로 확인: BA가 0.94 -> 0.87 로 급락).

    이를 막기 위해 예측 시점에 `보정된 확률 / 학습 시 클래스 사전확률` 로
    나눠 다시 argmax를 취하는 표준적인 prior-correction(사전확률 보정)을
    적용합니다. predict_proba()는 해석용으로 보정된 원본 확률을 그대로
    반환하고, predict()만 이 보정을 적용해 Balanced Accuracy를 지킵니다.
    """

    def __init__(self, calibrated_estimator, class_prior: dict):
        self.calibrated_estimator = calibrated_estimator
        self.class_prior = class_prior

    def fit(self, X, y):
        # 이미 학습된 calibrated_estimator를 감싸는 용도이므로 별도 학습 없음
        self.classes_ = self.calibrated_estimator.classes_
        self._prior_vec = np.array([self.class_prior[c] for c in self.classes_])
        return self

    def predict_proba(self, X):
        return self.calibrated_estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        adjusted = proba / self._prior_vec
        idx = adjusted.argmax(axis=1)
        return self.classes_[idx]


def get_candidate_models(n_jobs: int = -1) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "HistGradientBoost": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }

    # 선택적 의존성: 설치되어 있으면 후보에 포함 (requirements.txt에 명시됨)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = LabelEncodedClassifier(XGBClassifier(
            n_estimators=300, tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=n_jobs,
            eval_metric="mlogloss",
        ))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, random_state=RANDOM_STATE,
            auto_class_weights="Balanced", verbose=False,
        )
    except ImportError:
        pass

    return models
