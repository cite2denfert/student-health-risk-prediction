"""전처리 파이프라인 구성 (모델링 파이프라인 7단계 중 ①~③)"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import add_derived_features


def numeric_feature_names() -> list:
    return config.RAW_NUMERIC_COLS + config.DERIVED_FEATURE_NAMES


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric  : Median Imputation -> StandardScaler
    Categorical : Most Frequent Imputation -> One-Hot Encoding
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_feature_names()),
            ("cat", categorical_pipeline, config.RAW_CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df):
    """원본 df에 파생변수를 추가하고, 모델 입력에 필요한 컬럼만 반환합니다."""
    enriched = add_derived_features(df)
    keep_cols = numeric_feature_names() + config.RAW_CATEGORICAL_COLS
    keep_cols = [c for c in keep_cols if c in enriched.columns]
    return enriched[keep_cols]
